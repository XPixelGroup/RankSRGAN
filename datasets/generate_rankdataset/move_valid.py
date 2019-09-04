import os, random, shutil
Level1_patchsave_path = '/home/wlzhang/RankSRGAN/data/Rank_dataset_test/DF2K_train_patch_esrgan/';
Level2_patchsave_path = '/home/wlzhang/RankSRGAN/data/Rank_dataset_test/DF2K_train_patch_srgan/';
Level3_patchsave_path = '/home/wlzhang/RankSRGAN/data/Rank_dataset_test/DF2K_train_patch_srres/';

Level1_valid_patchsave_path = '/home/wlzhang/RankSRGAN/data/Rank_dataset_test/DF2K_valid_patch_esrgan/';
Level2_valid_patchsave_path = '/home/wlzhang/RankSRGAN/data/Rank_dataset_test/DF2K_valid_patch_srgan/';
Level3_valid_patchsave_path = '/home/wlzhang/RankSRGAN/data/Rank_dataset_test/DF2K_valid_patch_srres/';

if not os.path.exists(Level1_valid_patchsave_path):
	os.makedirs(Level1_valid_patchsave_path)
else:
	print('exists')

if not os.path.exists(Level2_valid_patchsave_path):
	os.makedirs(Level2_valid_patchsave_path)
else:
	print('exists')

if not os.path.exists(Level3_valid_patchsave_path):
	os.makedirs(Level3_valid_patchsave_path)
else:
	print('exists')

pathDir = os.listdir(Level1_patchsave_path)    #取图片的原始路径
filenumber=len(pathDir)
rate=0.1
picknumber=int(filenumber*rate)

sample = random.sample(pathDir, picknumber)


for name in sample:

	name = "".join(name)
	name = name.split('_')
	print(name[0])

	shutil.move(Level1_patchsave_path+name[0]+'_esrgan.png', Level1_valid_patchsave_path)
	shutil.move(Level2_patchsave_path+name[0]+'_srgan.png', Level2_valid_patchsave_path)
	shutil.move(Level3_patchsave_path+name[0]+'_srres.png', Level3_valid_patchsave_path)
