
% viewpoint annotation path
%path_ann_view = '../Annotations';
path_ann_view = fullfile('data/pascal3D', 'Annotations');

% read ids of validation images
addpath(fullfile('data/pascal3D', 'VDPM'));
addpath(fullfile('data/pascal3D', 'PASCAL/VOCdevkit/VOCcode'));
VOCinit;
%pascal_init;
ids = textread(sprintf(VOCopts.imgsetpath, 'val'), '%s');
M = numel(ids);

for i = 1:M
    % read ground truth bounding box
    rec = PASreadrecord(sprintf(VOCopts.annopath, ids{i}));
    voc12val_records{i} = rec;
end

save(fullfile('data/pascal3D', 'voc12val_records.mat'),'voc12val_records');
