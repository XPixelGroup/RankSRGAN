function sharpness = compute_sharpness(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap)

window = fspecial('gaussian',7,7/6);
window = window/sum(sum(window));

sigma = sqrt(imfilter((im-imfilter(im,window,'replicate')).*(im-imfilter(im,window,'replicate')),window,'replicate'));
paper_sharpness = blkproc(sigma,[blocksizerow blocksizecol],[blockrowoverlap blockcoloverlap],@computemean);
paper_sharpness = paper_sharpness(:);
sharpness = sum(paper_sharpness);









end
