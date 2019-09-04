function img_patch=save_patch_img(file_path,file_name,...
                    save_dir,save_name,h,w,patch_sz)

img = imread(fullfile(file_path,file_name));
img_patch = img(h:h+patch_sz-1,w:w+patch_sz-1,:);

imwrite(img_patch,fullfile(save_dir,save_name));

end
