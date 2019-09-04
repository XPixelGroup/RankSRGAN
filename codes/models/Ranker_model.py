import os
from collections import OrderedDict
import logging

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.nn.parallel import DataParallel, DistributedDataParallel

import models.networks as networks
from .base_model import BaseModel

logger = logging.getLogger('base')


class Ranker_Model(BaseModel):
    def name(self):
        return 'Ranker_Model'

    def __init__(self, opt):
        super(Ranker_Model, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define networks and load pretrained models
        self.netR = networks.define_R(opt).to(self.device)
        if opt['dist']:
            self.netR = DistributedDataParallel(self.netR, device_ids=[torch.cuda.current_device()])
        else:
            self.netR = DataParallel(self.netR)
        self.load()

        if self.is_train:
            self.netR.train()

            # loss
            self.RankLoss = nn.MarginRankingLoss(margin=0.5)
            self.RankLoss.to(self.device)
            self.L2Loss = nn.L1Loss()
            self.L2Loss.to(self.device)
            # optimizers
            self.optimizers = []
            wd_R = train_opt['weight_decay_R'] if train_opt['weight_decay_R'] else 0
            optim_params = []
            for k, v in self.netR.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    print('WARNING: params [%s] will not optimize.' % k)
            self.optimizer_R = torch.optim.Adam(optim_params, lr=train_opt['lr_R'], weight_decay=wd_R)
            print('Weight_decay:%f' % wd_R)
            self.optimizers.append(self.optimizer_R)

            # schedulers
            self.schedulers = []
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                                                                    train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        print('---------- Model initialized ------------------')
        self.print_network()
        print('-----------------------------------------------')

    def feed_data(self, data, need_img2=True):
        # input img1
        self.input_img1 = data['img1'].to(self.device)

        # label score1
        self.label_score1 = data['score1'].to(self.device)

        if need_img2:
            # input img2
            self.input_img2 = data['img2'].to(self.device)

            # label score2
            self.label_score2 = data['score2'].to(self.device)

            # rank label
            self.label = self.label_score1 >= self.label_score2  # get a ByteTensor
            # transfer into FloatTensor
            self.label = self.label.float()
            self.label = (self.label - 0.5) * 2


    def optimize_parameters(self, step):
        self.optimizer_R.zero_grad()
        self.predict_score1 = self.netR(self.input_img1)
        self.predict_score2 = self.netR(self.input_img2)


        self.predict_score1 = torch.clamp(self.predict_score1, min=-5, max=5)
        self.predict_score2 = torch.clamp(self.predict_score2, min=-5, max=5)

        l_rank = self.RankLoss(self.predict_score1, self.predict_score2, self.label)

        l_rank.backward()
        self.optimizer_R.step()

        # set log
        self.log_dict['l_rank'] = l_rank.item()

    def test(self):
        self.netR.eval()
        self.predict_score1 = self.netR(self.input_img1)
        self.netR.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()  # ............................
        out_dict['predict_score1'] = self.predict_score1.data[0].float().cpu()

        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netR)
        if isinstance(self.netR, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netR.__class__.__name__,
                                             self.netR.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netR.__class__.__name__)
        logger.info('Network R structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def load(self):

        load_path_R = self.opt['path']['pretrain_model_R']
        if load_path_R is not None:
            logger.info('Loading pretrained model for R [{:s}] ...'.format(load_path_R))
            self.load_network(load_path_R, self.netR)
    def save(self, iter_step):
        self.save_network(self.netR, 'R', iter_step)
