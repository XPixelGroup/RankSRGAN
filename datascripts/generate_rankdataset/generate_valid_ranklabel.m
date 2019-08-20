
%% set path
start = clock;
data_path = '../../data/Rank_dataset_test/';

level1_path = [data_path,'DF2K_valid_patch_esrgan'];
level2_path = [data_path,'DF2K_valid_patch_srgan'];
level3_path = [data_path,'DF2K_valid_patch_srres'];

ranklabel_path = [data_path,'/DF2K_train_NIQE.txt'];

level1_dir = fullfile(pwd,level1_path);
level2_dir = fullfile(pwd,level2_path);
level3_dir = fullfile(pwd,level3_path);

% Number of pixels to shave off image borders when calcualting scores
shave_width = 4;

% Set verbose option
verbose = true;
%% Calculate scores and save
addpath utils
addpath(genpath(fullfile(pwd,'utils')));

%% Reading file list
level1_file_list = dir([level1_dir,'/*.png']);
level2_file_list = dir([level2_path,'/*.png']);
level3_file_list = dir([level3_path,'/*.png']);

im_num = length(level1_file_list)
%fprintf(' %f\n',im_num);

%% Calculating scores
txtfp = fopen(ranklabel_path,'w');
    tic;
pp = parpool('local',28);
pp.IdleTimeout = 9800
disp('Already initialized'); %Strating
fprintf('-------- Strating -----------');
parfor ii=(1:im_num)
	[scoresname,scoresniqe] = parcal_niqe(ii,level3_dir,level3_file_list,im_num)
	level3_name{ii} = scoresname;
	level3_niqe(ii) = scoresniqe;

end
parfor ii=(1:im_num)
	[scoresname,scoresniqe] = parcal_niqe(ii,level2_dir,level2_file_list,im_num)
	level2_name{ii} = scoresname;
	level2_niqe(ii) = scoresniqe;
end
parfor ii=(1:im_num)
	[scoresname,scoresniqe] = parcal_niqe(ii,level1_dir,level1_file_list,im_num)
	level1_name{ii} = scoresname;
	level1_niqe(ii) = scoresniqe;
end

    toc;
delete(pp)
txtfp = fopen(ranklabel_path,'w');
for ii=(1:im_num)
    fprintf(txtfp,level3_name{ii});
    fprintf(txtfp,' %f\n',level3_niqe(ii));
    fprintf(txtfp,level2_name{ii});
    fprintf(txtfp,' %f\n',level2_niqe(ii));
    fprintf(txtfp,level1_name{ii});
    fprintf(txtfp,' %f\n',level1_niqe(ii));
end

fclose(txtfp);
