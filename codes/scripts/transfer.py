import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from collections import OrderedDict
import torch
import torch.nn as nn
import block as B
import os
import math
import functools
import arch_util as arch_util

class SRResNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4, norm_type= None , act_type='relu', \
            mode='CNA', res_scale=1, upsample_mode='pixelshuffle'):
        super(SRResNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        resnet_blocks = [B.ResNetBlock(nf, nf, nf, norm_type=norm_type, act_type=act_type,\
            mode=mode, res_scale=res_scale) for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*resnet_blocks, LR_conv)),\
            *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x
class mmsrSRResNet(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(mmsrSRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.recon_trunk = arch_util.make_layer(basic_block, nb)
        self.LRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        arch_util.initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last],
                                     0.1)
        if self.upscale == 4:
            arch_util.initialize_weights(self.upconv2, 0.1)

    def forward(self, x):
        fea = self.conv_first(x)
        out = self.recon_trunk(fea)
        out = self.LRconv(out) 

        if self.upscale == 4:
            out = self.relu(self.pixel_shuffle(self.upconv1(out+fea)))
            out = self.relu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.relu(self.pixel_shuffle(self.upconv1(out+fea)))

        out = self.conv_last(self.relu(self.HRconv(out)))

        return out


def transfer_network(load_path, network,ordereddict, strict=True):
    load_net = torch.load(load_path)
    load_net_dict = OrderedDict()  # remove unnecessary 'module.'
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    load_model_key = []
    for k, v in load_net.items():
        load_net_dict[k] = v
        load_model_key.append(k)

    i = 0
    for param_tensor in model2.state_dict():
            load_net_clean[param_tensor] = load_net_dict[load_model_key[i]]
            print('-------')
            print(param_tensor)
            print(load_model_key[i])
            i=i+1
    print(i)

    torch.save(load_net_clean, '/home/wlzhang/mmsr/experiments/pretrained_models/mmsr_SRResNet_pretrain.pth')
    network.load_state_dict(load_net_clean, strict=strict)

net_old = SRResNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = net_old.to(device)

# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

net_new = mmsrSRResNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model2 = net_new.to(device)

print("Model2's state_dict:")
ordereddict = []
for param_tensor in model2.state_dict():
    ordereddict.append(param_tensor)
    # print(param_tensor, "\t", model2.state_dict()[param_tensor].size())

# print("key state_dict:")
# print(ordereddict)

transfer_network('/home/wlzhang/mmsr/experiments/pretrained_models/SRResNet_bicx4_in3nf64nb16.pth', net_new, ordereddict)


# print("key:")
# print(ordereddict)
# summary(model, (3, 296, 296))