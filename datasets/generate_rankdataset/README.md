## How To Prepare Rank Dataset
### Prepare perceptual data
1. Prepare Three levels SR Models. You can download the [SRResNet (SRResNet_bicx4_in3nf64nb16.pth), 
SRGAN (SRGAN.pth), ESRGAN (ESRGAN_SuperSR.pth)] from 
[Google Drive](https://drive.google.com/drive/folders/16DkwrBa4cIqAoTbGU_bKMYoATcXC4IT6?usp=sharing) 
or [Baidu Drive](https://pan.baidu.com/s/1HFZokeAWne9oUkmJBnGr-A). You could place them in [`./experiments/pretrained_models/`](../../master/experiments/pretrained_models/).

2. Download [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flickr2K](https://github.com/LimBee/NTIRE2017)
from [Google Drive](https://drive.google.com/drive/folders/1B-uaxvV9qeuQ-t7MFiN1oEdA6dKnj2vW?usp=sharing) or
[Baidu Drive](https://pan.baidu.com/s/1CFIML6KfQVYGZSNFrhMXmA)
3. Generate Three level images using 'How to test' with [`codes/options/test/test_RankSRGAN.yml`](../../master/codes/options/test/test_RankSRGAN.yml) 
### Generate rank dataset
4. **Training dataset:** Use  [`./datasets/generate_rankdataset/generate_rankdataset.m`](../../master/datasets/generate_rankdataset/generate_rankdataset.m)
   to generate three level training patchs.
5. **Validation dataset:** Use  [`./datasets/generate_rankdataset/move_valid.py`](../../master/datasets/generate_rankdataset/move_valid.py)
   to generate three level patchs.
6. **Rank label:** Use  [`./datasets/generate_rankdataset/generate_train_ranklabel.m`](../../master/datasets/generate_rankdataset/generate_train_ranklabel.m)
   to generate Training Rank label (NIQE). 
   Use  [`./datasets/generate_rankdataset/generate_valid_ranklabel.m`](../../master/datasets/generate_rankdataset/generate_valid_ranklabel.m)
   to generate Validation Rank label (NIQE). 
   
   
