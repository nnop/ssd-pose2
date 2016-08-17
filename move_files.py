import glob
import os
import os.path as osp
import shutil


def main():
    log_dir = '/home/poirson/logs/'
    mod_temp = 'models/VGGNet/stash/'
    mod_dir = 'models/VGGNet/Pascal3D/'

    models = os.listdir(mod_temp)

    log_fi = glob.glob(osp.join(log_dir, '*.log'))

    for fi in log_fi:
        fidx = fi.strip('.log')
        #fidx = fidx.strip(osp.join(log_dir, 'VGG_Pascal3D_'))
        # hack
        fidx = fidx[len(osp.join(log_dir, 'VGG_Pascal3D_')):]

        if fidx in models:
            shutil.move(osp.join(mod_temp, fidx), mod_dir)

    models = os.listdir(mod_dir)

    for fi in log_fi:
        fidx = fi.strip('.log')
        #print fidx
        
        #fidx = fidx.strip(osp.join(log_dir, 'VGG_Pascal3D_'))
        # hack
        fidx = fidx[len(osp.join(log_dir, 'VGG_Pascal3D_')):]
        #print fidx

        if fidx not in models:
            print fidx

    # cleaning up files
    
    for m in models:
        cor_log = osp.join(log_dir, 'VGG_Pascal3D_' + m + '.log')
        if cor_log not in log_fi:
            shutil.move(osp.join(mod_dir, m), mod_temp)
    


if __name__ == '__main__':
    main()

