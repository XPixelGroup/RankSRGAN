function [scoresname,scoresNIQE] = parcal_niqemse(ii,input_dir,file_list,im_num)
   
    load modelparameters.mat
    fprintf(['\nCalculating scores for image ',num2str(ii),' / ',num2str(im_num)]);
    % Reading and converting images
    input_image_path = fullfile(input_dir,file_list(ii).name);
    input_image = convert_shave_image(imread(input_image_path),4);

    % Calculating scores
    scoresname = file_list(ii).name;

    scoresNIQE = computequality(input_image,96,96,...
        0,0,mu_prisparam,cov_prisparam);

    fprintf([' perceptual scores is: ',num2str(scoresNIQE),' ',scoresname,'']);
end