## Place pretrained models here. 

### Pretrained models

1. `SRResNet_bicx4_in3nf64nb16.pth`: the well-trained SRResNet model in PSNR orientation. 
2. `SRGAN.pth`: the pretrained model SRGAN implemented by [BasicSR](https://github.com/xinntao/BasicSR). 
3. `ESRGAN_SuperSR.pth`: the pretrained model [ESRGAN_SuperSR](https://github.com/xinntao/ESRGAN). 

### Three pretrained Ranker models :
1. `Ranker_NIQE.pth`: the well-trained Ranker with **NIQE** metric. 
2. `Ranker_Ma.pth`: the well-trained Ranker with **Ma** metric. 
3. `Ranker_PI.pth`: the well-trained Ranker with **PI** metric. 

### RankSRGAN models
1. `RankSRGAN_NIQE.pth`: the RankSRGAN in **NIQE** orientation. 
2. `RankSRGAN_Ma.pth`: the RankSRGAN in **Ma** orientation. 
3. `RankSRGAN_PI.pth`: the RankSRGAN in **PI** orientation. 



*Note that* the pretrained models are trained under the `MATLAB bicubic` kernel. 
If the downsampled kernel is different from that, the results may have artifacts.
