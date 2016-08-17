import argparse
import glob
import json
import os
import os.path as osp
import shutil

'''
This script removes models which have been run trained for less
than some number of iterations
'''



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove old / unused models ')
    parser.add_argument('--iter', default=0, type=int, help='minimum number of iterations')
    args = parser.parse_args()
    args = vars(args)
    
    mod_path = 'models/VGGNet/Pascal3D/'
    models = os.listdir(mod_path)

    for mod in models:
        if mod == 'stash':
            continue
        mod_dir = osp.join(mod_path, mod)
        caf_mods = glob.glob(osp.join(mod_dir, '*.caffemodel'))
        found = False

        for caf in caf_mods:
            temp = caf.strip('.caffemodel')
            stidx = temp.find('iter_')
            stidx += len('iter_')
            iterx = int(temp[stidx:])

            if iterx > args['iter']:
                found = True
                break

        # try and remove model / logs / options
        if not found:
            #shutil.rmtree(mod_dir)
            print 'removed %s' % mod_dir

            log_file = osp.join('/home/poirson/logs/', 'VGG_Pascal3D_' + mod + '.log' )
            if osp.exists(log_file):
                #os.remove(log_file)
                print 'removed %s' % log_file

            if 'SSD_share_pose_' in mod:
                stidx = mod.find('SSD_share_pose_')
                stidx += len('SSD_share_pose_')
                try:
                    idx = int(mod[stidx:])
                except:
                    idx = 0
            elif 'SSD_seperate_pose_' in mod:
                stidx = mod.find('SSD_seperate_pose_')
                stidx += len('SSD_seperate_pose_')
                try:
                    idx = int(mod[stidx:])
                except:
                    idx = 0
            else:
                idx = 0

            # try to remove options file
            opt_file = osp.join('/home/poirson/options/', '%d.json' % idx)
            if osp.exists(opt_file):
                #os.remove(opt_file)
                print 'removed %s' % opt_file


