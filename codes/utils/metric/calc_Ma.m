function score = calc_Ma(input_image_path_list)
addpath(genpath(fullfile(pwd,'utils')));
Ma = 0;
NIQE = 0;
PI = 0;

pp = parpool('local',28);
pp.IdleTimeout = 9800;
parfor ii=(1:100)
	[Mascores,NIQEscores,PIscores] = parcalMa(ii,input_image_path_list);
	Ma = Ma + Mascores;
	NIQE = NIQE + NIQEscores;
	PI = PI + PIscores;

end

score.Ma = Ma;
score.NIQE = NIQE;
score.PI = PI;

delete(pp)

end

