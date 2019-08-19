function [Mascores,NIQEscores,PIscores] = parcalMa(ii,input_image_path_list)
	%% Loading model
	shave_width = 4;
	load modelparameters.mat
	blocksizerow    = 96;
	blocksizecol    = 96;
	blockrowoverlap = 0;
	blockcoloverlap = 0;
	%% Calculating scores
	input_image_path = input_image_path_list{ii};

    input_image = convert_shave_image(imread(input_image_path),shave_width);
	        
    % Calculating scores
    NIQEscores = computequality(input_image,blocksizerow,blocksizecol,...
        blockrowoverlap,blockcoloverlap,mu_prisparam,cov_prisparam);
    Mascores = quality_predict(input_image);
    PIscores = (NIQEscores + (10 - Mascores)) / 2;

end