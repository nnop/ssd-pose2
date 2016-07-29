close all;
classes = {'bicycle',
               'motorbike',
               'car',
               'bus',
               'train',
              'diningtable',
              'aeroplane',
              'sofa',
              'tvmonitor',
              'chair',
              'boat'};

          
aps = [];

bins = 8;

for i=1:length(classes)
    cls = classes{i};
    fname = [cls '.mat'];
    
    [recall, precision, accuracy, ap, aa] = compute_recall_precision_accuracy_azimuth(cls, bins, bins, fname);
    %ap = 0;
    disp(ap)
    aps = [aps ap];
end

disp(aps)
mean(aps)

