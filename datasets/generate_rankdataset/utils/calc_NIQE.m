function NIQE = calc_NIQE(input_image)

addpath(genpath(fullfile(pwd,'utils')));

%% Loading model
load modelparameters.mat
blocksizerow    = 96;
blocksizecol    = 96;
blockrowoverlap = 0;
blockcoloverlap = 0;

%% Reading file list
%scores = struct([]);


    % Calculating scores
    NIQE = computequality(input_image,blocksizerow,blocksizecol,...
        blockrowoverlap,blockcoloverlap,mu_prisparam,cov_prisparam);
%     perceptual_score = ([scores(ii).NIQE] + (10 - [scores(ii).Ma])) / 2;
%    perceptual_score = scores(ii).NIQE;
%    fprintf([' perceptual scores is: ',num2str(perceptual_score),' ',scores(ii).name,'']);

end
