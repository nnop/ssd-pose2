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
        fi = fi.strip('.log')
        fi = fi.strip(osp.join(log_dir, 'VGG_Pascal3D_'))

        if fi in models:
            shutil.move(osp.join(mod_temp, fi), mod_dir)


if __name__ == '__main__':
    main()

