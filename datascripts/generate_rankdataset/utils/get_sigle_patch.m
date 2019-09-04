function img_patch=get_sigle_patch(file_path,file_name,...
                    save_dir,save_name,h,w,patch_sz)

img = imread(fullfile(file_path,file_name));
img_patch = img(h:h+patch_sz-1,w:w+patch_sz-1,:);


end
