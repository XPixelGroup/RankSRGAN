# image SR training

python train.py -opt options/train/train_RankSRGAN.yml # Validation with PSNR
python train_niqe.py -opt options/train/train_RankSRGAN.yml # Validation with PSNR and NIQE

# Ranker training

python train_rank.py -opt options/train/train_Ranker.yml #
