
Dataloader

- use opencv (`cv2`) to read and process images.

- read from **image** files OR from **.lmdb** for fast IO speed.
    - How to create .lmdb file? Please see [`codes/scripts/create_lmdb.py`](../scripts/create_lmdb.py).
    
- can downsample images using `matlab bicubic` function. However, the speed is a bit slow. Implemented in [`util.py`](util.py). More about [`matlab bicubic` function](https://github.com/xinntao/BasicSR/wiki/Matlab-bicubic-imresize).


## Contents

- `LR_dataset`: only reads LR images in test phase where there is no GT images.
- `LRHR_dataset`: reads LR and HR pairs from image folder or lmdb files. If only HR images are provided, downsample the images on-the-fly. Used in SR and SRGAN training and validation phase.
- `Rank_IMIM_Pair_dataset`: reads rank data pairs from image folder.
## How To Prepare Data
### SR, SRGAN
1. Prepare the images. You can download **classical SR** datasets (including BSD200, T91, General100; Set5, Set14, urban100, BSD100, manga109; historical) from [Google Drive](https://drive.google.com/drive/folders/1pRmhEmmY-tPF7uH8DuVthfHoApZWJ1QU?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/18fJzAHIg8Zpkc-2seGRW4Q). DIV2K dataset can be downloaded from [DIV2K offical page](https://data.vision.ee.ethz.ch/cvl/DIV2K/), or from [Baidu Drive](https://pan.baidu.com/s/1LUj90_skqlVw4rjRVeEoiw).

1. For faster IO speed, you can make lmdb files for training dataset. Please see [`codes/scripts/create_lmdb.py`](../scripts/create_lmdb.py).

1. We use DIV2K dataset for training the SR and SRGAN models. 
    1. since DIV2K images are large, we first crop them to sub images using [`codes/scripts/extract_subimgs_single.py`](../scripts/extract_subimgs_single.py). 
    1. generate LR images using matlab with [`codes/scripts/generate_mod_LR_bic.m`](../scripts/generate_mod_LR_bic.m). If you already have LR images, you can skip this step. Please make sure the LR and HR folders have the same number of images.
    1. generate .lmdb file if needed using [`codes/scripts/create_lmdb.py`](../scripts/create_lmdb.py).
    1. modify configurations in `options/train/xxx.json` when training, e.g., `dataroot_HR`, `dataroot_LR`.

### data augmentation

We use random crop, random flip/rotation, (random scale) for data augmentation. 

### wiki

[Color-conversion-in-SR](https://github.com/xinntao/BasicSR/wiki/Color-conversion-in-SR)


<!--## TODO

- [ ] verify random scale
-->
