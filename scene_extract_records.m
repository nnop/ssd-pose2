function []=scene_extract_records(scene_path)
    % viewpoint annotation path
    %path_ann_view = '../Annotations';
    %path_ann_view = fullfile('data/pascal3D', 'Annotations');
    data_path = 'data/scenes/matTest/';
    %scene_path = 'test_bins=8_scene=1';

    % read ids of validation images
    %addpath(fullfile('data/pascal3D', 'VDPM'));
    %addpath(fullfile('data/pascal3D', 'PASCAL/VOCdevkit/VOCcode'));
    %VOCinit;
    %pascal_init;
    %ids = textread(fullfile(data_path, scene_path, 'jsons.txt'), '%s');
    ids = textscan(fopen(fullfile(data_path, scene_path, 'jsons.txt')), '%s', 'whitespace', '\n');
    ids = ids{1};

    M = length(ids);

    for i = 1:M
        % read ground truth bounding box
        idx = ids(i);
        rec = load(char(fullfile(data_path, scene_path, idx)));
        voc12val_records{i} = rec;
    end

    save(fullfile(data_path, scene_path, 'scene_records.mat'),'voc12val_records');

end