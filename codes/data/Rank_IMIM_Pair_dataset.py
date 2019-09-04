import os.path
import random
import cv2
import numpy as np
import torch
import torch.utils.data as data
import data.util as util
from itertools import combinations
from scipy.special import comb


class RANK_IMIM_Pair_Dataset(data.Dataset):
    '''
    Read LR and HR image pair.
    If only HR image is provided, generate LR image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def name(self):
        return 'RANK_IMIM_Pair_Dataset'

    def __init__(self, opt, is_train):
        super(RANK_IMIM_Pair_Dataset, self).__init__()
        self.opt = opt

        self.is_train = is_train

        # read image list from lmdb or image files

        self.paths_img1, self.sizes_GT = util.get_image_paths(opt['data_type'], opt['dataroot_img1'])
        self.paths_img2, self.sizes_GT = util.get_image_paths(opt['data_type'], opt['dataroot_img2'])
        self.paths_img3, self.sizes_GT = util.get_image_paths(opt['data_type'], opt['dataroot_img3'])

        self.img_env1 = None
        self.img_env2 = None
        self.img_env3 = None

        self.label_path = opt['dataroot_label_file']

        # get image label scores
        self.label = {}
        f = open(self.label_path, 'r')
        for line in f.readlines():
            line = line.strip().split()
            self.label[line[0]] = line[1]
        f.close()

        assert self.paths_img1, 'Error: img1 paths are empty.'

        # self.random_scale_list = [1, 0.9, 0.8, 0.7, 0.6, 0.5]
        self.random_scale_list = None

    def __getitem__(self, index):

        if self.is_train:
            # get img1 and img1 label score
            # choice = random.choice(['img1_img2','img1_img2','img1_img2','img1_img3','img2_img3']) #Oversampling for hard sample
            choice = random.choice(['img1_img2', 'img1_img3', 'img2_img3'])

            # print(choice)

            if choice == 'img1_img2':
                img1_path = self.paths_img1[index]
                img1 = util.read_img(self.img_env1, img1_path)
                img2_path = self.paths_img2[index]
                img2 = util.read_img(self.img_env2, img2_path)
            elif choice == 'img1_img3':
                img1_path = self.paths_img1[index]
                img1 = util.read_img(self.img_env1, img1_path)
                img2_path = self.paths_img3[index]
                img2 = util.read_img(self.img_env3, img2_path)

            elif choice == 'img2_img3':
                img1_path = self.paths_img2[index]
                img1 = util.read_img(self.img_env2, img1_path)
                img2_path = self.paths_img3[index]
                img2 = util.read_img(self.img_env3, img2_path)


            img1_name = img1_path.split('/')[-1]
            img1_score = np.array(float(self.label[img1_name]), dtype='float')
            img1_score = img1_score.reshape(1)

            img2_name = img2_path.split('/')[-1]
            img2_score = np.array(float(self.label[img2_name]), dtype='float')
            img2_score = img2_score.reshape(1)

            if img1.shape[2] == 3:
                img1 = img1[:, :, [2, 1, 0]]
            img1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img1, (2, 0, 1)))).float()
            img1_score = torch.from_numpy(img1_score).float()

            if img2.shape[2] == 3:
                img2 = img2[:, :, [2, 1, 0]]
            img2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img2, (2, 0, 1)))).float()
            img2_score = torch.from_numpy(img2_score).float()

            # print('img1:'+img1_name,' & ','img2:'+img2_name)

        else:
            # get img1
            img1_path = self.paths_img1[index]
            img1 = util.read_img(self.img_env1, img1_path)

            img1_name = img1_path.split('/')[-1]
            img1_score = np.array(float(self.label[img1_name]), dtype='float')
            img1_score = img1_score.reshape(1)

            if img1.shape[2] == 3:
                img1 = img1[:, :, [2, 1, 0]]
            img1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img1, (2, 0, 1)))).float()
            img1_score = torch.from_numpy(img1_score).float()
            # print('img1:'+img1_name)

            # not useful
            img2_path = img1_path
            img2 = img1
            img2_score = img1_score

        # exit()

        return {'img1': img1, 'img2': img2, 'img1_path': img1_path, 'img2_path': img2_path, 'score1': img1_score,
                'score2': img2_score}

    def __len__(self):
        return len(self.paths_img1)
