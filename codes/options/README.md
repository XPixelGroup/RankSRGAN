# Configurations
- Use **json** files to configure options.
- Convert the json file to python dict.
- Support `//` comments and use `null` for `None`.

## Table
Click for detailed explanations for each json file.

1. [RankSRGAN_NIQE.json](#RankSRGAN_NIQE_json)
2. [Ranker.json](#Ranker_json) 

## RankSRGAN_NIQE_json
```c++
{
  "name": "RankSRGANx4_NIQE" 
  ,"model":"ranksrgan" // use tensorboard_logger
  ,"scale": 4
  ,"gpu_ids": [2] // specify GPUs, actually it sets the `CUDA_VISIBLE_DEVICES`

  ,"datasets": {  // configure the training and validation datasets
    "train": { // training dataset configurations
      "name": "DIV2K"
      ,"mode": "LRHR"
      ,"dataroot_HR": "/home/wlzhang/BasicSR12/data/DIV2K800_sub.lmdb" // HR data root
      ,"dataroot_LR": "/home/wlzhang/BasicSR12/data/DIV2K800_sub_bicLRx4.lmdb" // LR data root
      ,"subset_file": null
      ,"use_shuffle": true
      ,"n_workers": 8  // number of data load workers
      ,"batch_size": 8
      ,"HR_size": 296 // 128 for SRGAN | 296 for RankSRGAN, cropped HR patch size
      ,"use_flip": true
      ,"use_rot": true
      , "random_flip": false // whether use horizontal and vertical flips
      , "random_scale": false // whether use rotations: 90, 190, 270 degrees
    }
    , "val": { // validation dataset configurations
      "name": "val_PIRM"
      ,"mode": "LRHR"
      ,"dataroot_HR": "/home/wlzhang/BasicSR12/data/val/PIRMtestHR"
      ,"dataroot_LR": "/home/wlzhang/BasicSR12/data/val/PIRMtest"
    }
  }

  ,"path": {
    "root": "/home/wlzhang/RankSRGAN", // root path
    // "resume_state": "../experiments/RankSRGANx4_NIQE/training_state/152000.state", // Resume the training from 152000 iteration
    "pretrain_model_G": "/home/wlzhang/RankSRGAN/experiments/pretrained_models/SRResNet_bicx4_in3nf64nb16.pth", // G network pretrain model
    "pretrain_model_R": "/home/wlzhang/RankSRGAN/experiments/pretrained_models/Ranker_NIQE.pth", // R network pretrain model

    "experiments_root": "/home/wlzhang/RankSRGAN/experiments/RankSRGANx4_NIQE",
    "models": "/home/wlzhang/RankSRGAN/experiments/RankSRGANx4_NIQE/models",
    "log": "/home/wlzhang/RankSRGAN/experiments/RankSRGANx4_NIQE",
    "val_images": "/home/wlzhang/RankSRGAN/experiments/RankSRGANx4_NIQE/val_images"
  }

  ,"network_G": { // configurations for the network G
    "which_model_G": "sr_resnet"
    ,"norm_type": null // null | "batch", norm type 
    ,"mode": "CNA" // Convolution mode: CNA for Conv-Norm_Activation
    ,"nf": 64 // number of features for each layer
    ,"nb": 16 // number of blocks
    ,"in_nc": 3 // input channels
    ,"out_nc": 3 // output channels
    ,"group": 1
  }
  ,"network_D": { // configurations for the network D
    "which_model_D": "discriminator_vgg_128"
    ,"norm_type": "batch"
    ,"act_type": "leakyrelu"
    ,"mode": "CNA"
    ,"nf": 64
    ,"in_nc": 3
  },
    "network_R": {
    "which_model_R": "Ranker_VGG12",
    "norm_type": "batch",
    "act_type": "leakyrelu",
    "mode": "CNA",
    "nf": 64,
    "in_nc": 3
  },
"train": { // training strategies
    "lr_G": 0.0001, // initialized learning rate for G
    "train_D": 1,
    "weight_decay_G": 0,
    "beta1_G": 0.9,
    "lr_D": 0.0001, // initialized learning rate for D
    "weight_decay_D": 0,
    "beta1_D": 0.9,
    "lr_scheme": "MultiStepLR", // learning rate decay scheme
    "lr_steps": [
      50000,
      100000,
      200000,
      300000
    ],
    "lr_gamma": 0.5,
    "pixel_criterion": "l1", // "l1" | "l2", pixel criterion
    "pixel_weight": 0,
    "feature_criterion": "l1", // perceptual criterion (VGG loss)
    "feature_weight": 1,
    "gan_type": "vanilla", // GAN type
    "gan_weight": 0.005,
    "D_update_ratio": 1,
    "D_init_iters": 0,
    "R_weight": 0.03, // Ranker-content loss
    "R_bias": 0,
    "manual_seed": 0,
    "niter": 500000.0, // total training iteration
    "val_freq": 5000 // validation frequency
  },
  "logger": { // logger configurations
    "print_freq": 200
    ,"save_checkpoint_freq": 5000
  },
  "timestamp": "180804-004247",
  "is_train": true,
  "fine_tune": false
}

```
## Ranker_json

```c++
{
  "name": "Ranker_NIQE" //  
  ,"use_tb_logger": true // use tensorboard_logger
  ,"model":"rank"
  ,"scale": 4
  ,"gpu_ids": [2,5] // specify GPUs, actually it sets the `CUDA_VISIBLE_DEVICES`
  ,"datasets": { // configure the training and validation rank datasets
    "train": { // training dataset configurations
      "name": "DF2K_train_rankdataset"
      ,"mode": "RANK_IMIM_Pair"
    ,"dataroot_HR": null
    ,"dataroot_LR":null
      ,"dataroot_img1": "/home/wlzhang/data/rankdataset/DF2K_train_patch_esrgan/" // Rankdataset: Perceptual level1 data root
      ,"dataroot_img2": "/home/wlzhang/data/rankdataset/DF2K_train_patch_srgan/" // Rankdataset: Perceptual level2 data root
    ,"dataroot_img3": "/home/wlzhang/data/rankdataset/DF2K_train_patch_srres/" // Rankdataset: Perceptual level3 data root
    ,"dataroot_label_file": "/home/wlzhang/data/rankdataset/DF2K_train_patch_label.txt" // Rankdataset: Perceptual rank label root
      ,"subset_file": null
      ,"use_shuffle": true
      ,"n_workers": 8 // number of data load workers
      ,"batch_size": 32 
      ,"HR_size": 128 
      ,"use_flip": true
      ,"use_rot": true
    }
    , "val": {
      "name": "DF2K_valid_rankdataset" // validation dataset configurations
      ,"mode": "RANK_IMIM_Pair"
    ,"dataroot_HR": null
    ,"dataroot_LR":null
      ,"dataroot_img1": "/home/wlzhang/data/rankdataset/DF2K_test_patch_all/"
    ,"dataroot_label_file": "/home/wlzhang/data/rankdataset/DF2K_test_patch_label.txt"
    }
  }

  ,"path": { // root path
    "root": "/home/wlzhang/RankSRGAN", 
    "experiments_root": "/home/wlzhang/RankSRGAN/experiments/Ranker_NIQE",
    "models": "/home/wlzhang/RankSRGAN/experiments/Ranker_NIQE/models",
    "log": "/home/wlzhang/RankSRGAN/experiments/Ranker_NIQE",
    "val_images": "/home/wlzhang/RankSRGAN/experiments/Ranker_NIQE/val_images"
  }

  ,"network_G": {
    "which_model_G": "sr_resnet"
    ,"norm_type": null
    ,"mode": "CNA"
    ,"nf": 64
    ,"nb": 16
    ,"in_nc": 3
    ,"out_nc": 3
    ,"group": 1
  }
  ,"network_R": { // configurations for the network Ranker
    "which_model_R": "Ranker_VGG12"
    ,"norm_type": "batch" // null | "batch", norm type 
    ,"act_type": "leakyrelu"
    ,"mode": "CNA"
    ,"nf": 64
    ,"nb": 16
    ,"in_nc": 3
    ,"out_nc": 3
    ,"in_nc": 3
  }

  ,"train": { // training strategies 
    "lr_R": 1e-3 // initialized learning rate for R
    ,"weight_decay_R": 1e-4
    ,"beta1_G": 0.9
    ,"lr_D": 1e-4
    ,"weight_decay_D": 0
    ,"beta1_D": 0.9
    ,"lr_scheme": "MultiStepLR"
    ,"lr_steps": [100000, 200000] // learning rate decay scheme

    ,"lr_gamma": 0.5

    // ,"pixel_criterion": "l1"
    // ,"pixel_weight": 1
    // ,"feature_criterion": "l1"
    // ,"feature_weight": 1
    // ,"gan_type": "vanilla"
    // ,"gan_weight": 5e-3

    ,"D_update_ratio": 1
    ,"D_init_iters": 0

    ,"manual_seed": 0 
    ,"niter": 400000 // total training iteration
    ,"val_freq": 5000 // validation frequency
  }
 
  ,"logger": { // logger configurations
    "print_freq": 200
    ,"save_checkpoint_freq": 5000
  }
}
'''
