
Level1_folder = '/home/wlzhang/data/DIV2K/DIV2K_train_ESRGAN/'
Level2_folder = '/home/wlzhang/data/DIV2K/DIV2K_train_srgan/'
Level3_folder = '/home/wlzhang/data/DIV2K/DIV2K_train_srres/'

Level1_filepaths = dir(fullfile(Level1_folder,'*.png'));
Level2_filepaths = dir(fullfile(Level2_folder,'*.png'));
Level3_filepaths = dir(fullfile(Level3_folder,'*.png'));

Level1_patchsave_path = '/home/wlzhang/RankSRGAN/data/Rank_dataset_test/DF2K_train_patch_esrgan';
Level2_patchsave_path = '/home/wlzhang/RankSRGAN/data/Rank_dataset_test/DF2K_train_patch_srgan';
Level3_patchsave_path = '/home/wlzhang/RankSRGAN/data/Rank_dataset_test/DF2K_train_patch_srres';

mkdir(Level3_patchsave_path)
mkdir(Level2_patchsave_path)
mkdir(Level1_patchsave_path)

addpath utils

patch_sz = 296;
stride = 148; %300 148 200
blocksizerow = 96;
blocksizecol = 96;
blockrowoverlap = 0;
blockcoloverlap = 0;

total_count = 0;
selected_count = 0;
display_flag = 0;
save_count = 0;
count_class = 0;
fprintf('-------- Strating -----------');
for k = 1 : length(Level1_filepaths)
    tic;
	fprintf('Processing img: %s\n',Level1_filepaths(k).name);
    fprintf('Processing srganimg: %s\n',Level2_filepaths(k).name);

	level1_img = imread(fullfile(Level1_folder,Level1_filepaths(k).name));
    level2_img = imread(fullfile(Level2_folder,Level2_filepaths(k).name));
    %srres_img = imread(fullfile(Level3_folder,srres_filepaths(k).name));
   
	if display_flag == 1
		subplot(1,2,1);
		imshow(level1_img);
	end
	img_height = size(level1_img,1);
	img_width = size(level1_img,2);
	
    h_num = ceil((img_height-patch_sz)/stride);
    w_num = ceil((img_width-patch_sz)/stride);
    count = 0;
    patch_list = zeros(patch_sz,patch_sz,3,1);

    % esrgan_patch_list1 = zeros(patch_sz,patch_sz,3,200); 
    % srres_patch_list1 = zeros(patch_sz,patch_sz,3,200); 
    %srgan_patch_list1 = zeros(patch_sz,patch_sz,3,200); 

    location_list = zeros(1,2,15);
    i = 0;
	for h = 1:stride:img_height-patch_sz
		for w = 1:stride:img_width-patch_sz
            count = count +1;
			total_count = total_count + 1;
			level1_img_patch = level1_img(h:h+patch_sz-1,w:w+patch_sz-1,:);
			if(size(level1_img_patch,3)==3)
				im = rgb2gray(level1_img_patch);
			end
			im = double(im);
			
			sharpness = compute_sharpness(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap);
			
			if sharpness >= 40
                i = i+1;
				if display_flag == 1
					subplot(1,2,1);
					rectangle('Position',[w,h,patch_sz,patch_sz],'edgecolor','r');
				end
				selected_count = selected_count + 1;
                level1_patch_list(:,:,:,i) = level1_img_patch;
                location_list(:,:,i) = [h,w];

			else
				if display_flag == 1
					subplot(1,2,1);
					rectangle('Position',[w,h,patch_sz,patch_sz],'edgecolor','k');
				end
			end
			
			
		end
    end
    fprintf('patch numbel:%d ',i);
    count1 = 0;
    count2 = 0;
    count3 = 0;
    for j = 1:i

        level1_save_patch = uint8(level1_patch_list(:,:,:,j));
        level1_NIQE = calc_NIQE(level1_save_patch);

        patch_h = location_list(1,1,j);
        patch_w = location_list(1,2,j);
        Level3_img_patch=get_sigle_patch(Level3_folder,Level1_filepaths(k).name,...
                        Level3_patchsave_path,...
                        [num2str(save_count) '_srres.png'],...
                        patch_h,patch_w,patch_sz);
        level2_img_patch=get_sigle_patch(Level2_folder,Level1_filepaths(k).name,...
                        Level2_patchsave_path,...
                        [num2str(save_count) '_srgan.png'],...
                        patch_h,patch_w,patch_sz);
        level2_NIQE = calc_NIQE(level2_img_patch);
        Level3_NIQE = calc_NIQE(Level3_img_patch);

        if abs(level2_NIQE - level1_NIQE) > 0.1
            count1 = count1+1;
            level1_patch_list1(:,:,:,count1) = level1_save_patch;
            level2_patch_list1(:,:,:,count1) = level2_img_patch;
            Level3_patch_list1(:,:,:,count1) = Level3_img_patch;
        end
      
    end
        fprintf('distance good:%d ',count1);
    if count1 < 200  %200
        for idx= 1:count1
            save_count = save_count + 1;
            
            save_name = [num2str(save_count) '_esrgan.png'];            
            level1_patch = uint8(level1_patch_list1(:,:,:,idx));		   
            imwrite(level1_patch,fullfile(Level1_patchsave_path,save_name));     
            
            save_name = [num2str(save_count) '_srgan.png'];            
            level2_patch = uint8(level2_patch_list1(:,:,:,idx));		   
            imwrite(level2_patch,fullfile(Level2_patchsave_path,save_name));
            
            save_name = [num2str(save_count) '_srres.png'];            
            Level3_patch = uint8(Level3_patch_list1(:,:,:,idx));		   
            imwrite(Level3_patch,fullfile(Level3_patchsave_path,save_name));
            
        end
    else
        rand_order = randperm(count1);
        for idx= 1:200
            save_count = save_count + 1;
            
            save_name = [num2str(save_count) '_esrgan.png'];            
            level1_patch = uint8(level1_patch_list1(:,:,:,rand_order(idx)));           
            imwrite(level1_patch,fullfile(Level1_patchsave_path,save_name));     
            
            save_name = [num2str(save_count) '_srgan.png'];            
            level2_patch = uint8(level2_patch_list1(:,:,:,rand_order(idx)));         
            imwrite(level2_patch,fullfile(Level2_patchsave_path,save_name));
            
            save_name = [num2str(save_count) '_srres.png'];            
            Level3_patch = uint8(Level3_patch_list1(:,:,:,rand_order(idx)));         
            imwrite(Level3_patch,fullfile(Level3_patchsave_path,save_name));
            
        end     
    end
    fprintf('Current save patches:%d\n',save_count);

    end
    toc;
fprintf('Total image patch: %d\n',total_count);
fprintf('Selected image patch: %d\n',selected_count);
fprintf('Generated image patch: %d\n',save_count);

    
	
	
