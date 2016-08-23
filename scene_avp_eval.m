close all;
classes = {'chair',
              'diningtable',
              'sofa',
              'tvmonitor'};

          
aps = [];
aas = [];


for i=1:length(classes)
    cls = classes{i};
    fname = fullfile(path, [cls '.mat']);
    
    [recall, precision, accuracy, ap, aa] = scene_compute_recall_precision_accuracy_azimuth(cls, bins, bins, fname, rotate, bin_fa);
    aas = [aas aa];
end


fid = fopen(fullfile(path, 'results.txt'), 'w');

for i=1:length(classes)
     fprintf(fid, sprintf('%.1f & ', aas(i)*100));
end
fprintf(fid, sprintf('%.1f\n', mean(aas)*100));
fprintf(fid, sprintf('%.1f\n', mean(aas)*100));

fclose(fid);

disp(mean(aas)*100);


