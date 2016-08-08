import argparse
import glob
import json
import os
import os.path as osp
import shutil
import subprocess

from utils import options

'''
This script removes models which have been run trained for less
than some number of iterations
'''

eval_dir = 'mat_eval/'

def main(args):
    mod_path = 'models/VGGNet/Pascal3D/'
    models = os.listdir(mod_path)

    eval_dir = 'mat_eval/'

    mod_to_opts = {}

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

            if iterx == args['iter']:
                found = True
                break

        # try and remove model / logs / options
        if found:
            # try to get options
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
                opt = options.Options(opt_file)
                mod_to_opts[mod] = opt

    for idx, val in mod_to_opts.iteritems():
        print idx
        print val.get_opts('rotate') 

    # make share and seperate table
    out_fi = open('share_sep_eval.txt', 'w')
    for des in [True, False]:
        # make output string
        out = ''
        if des:
            out += 'Share 300 &'
        else:
            out += 'Separate 300 &'

        for bins in [4, 8, 16, 24]:
            for mod, val in mod_to_opts.iteritems():
                if val.get_opts('share_pose') == des and\
                val.get_opts('num_bins') == bins and val.get_opts('rotate')\
                and val.get_opts('size') == 300:
                    out = make_out(mod, args['iter'], out)


        # write output
        out_fi.write(out + '\n')
    out_fi.close()


    # make with and without pascal table
    out_fi = open('no_pascal_eval.txt', 'w')
    for des in [True, False]:
        out = ''
        if des:
            out += 'Share 300 &'
        else:
            out += 'Share 300 Pascal &'

        for bins in [4, 8, 16, 24]:
            for mod, val in mod_to_opts.iteritems():
                if val.get_opts('share_pose') and val.get_opts('num_bins') == bins\
                and val.get_opts('rotate') and val.get_opts('imagenet') == des\
                and val.get_opts('size') == 300:
                    out = make_out(mod, args['iter'], out)

        out_fi.write(out + '\n')
    out_fi.close()


    # make rotated table
    out_fi = open('rot_eval.txt', 'w')
    for sz in [300, 500]:
        for rot in [False, True]:
            out = ''
            if rot:
                out += 'Share %d Rot &' % sz
            else:
                out += 'Share %d &' % sz

            for bins in [4, 8, 16, 24]:
                for mod, val in mod_to_opts.iteritems():
                    if val.get_opts('share_pose') and val.get_opts('num_bins') == bins\
                    and val.get_opts('rotate') == rot and val.get_opts('imagenet')\
                    and val.get_opts('size') == sz:
                        out = make_out(mod, args['iter'], out)
            out_fi.write(out + '\n')
    out_fi.close()

    
    # make large results table
    out_fi = open('all_res.txt', 'w')
    for bins in [4, 8, 16, 24]:
        out_fi.write('%d bins \n' % bins)
        for sz in [300, 500]:
            out = 'Ours Share %d &' % sz

            for mod, val in mod_to_opts.iteritems():
                if val.get_opts('share_pose') and val.get_opts('num_bins') == bins\
                and val.get_opts('rotate') and val.get_opts('imagenet')\
                and val.get_opts('size') == sz:
                    out = get_row(mod, args['iter'], out)

            out_fi.write(out + '\n')
    out_fi.close()


def get_row(mod, iterx, out):
    cmd = 'python runOfficialTest.py --model=%s --iter=%d' % (mod, iterx)
    subprocess.call(cmd, shell=True)

    # open output file
    eval_fi = osp.join(eval_dir, '%s_%d' % (mod, args['iter']), 'results.txt')
    with open(eval_fi, 'r') as f:
        lines = [line for line in f]
        # make output string
        out += ' %s &' % lines[0][:-1]

    return out   


def make_out(mod, iterx, out):
    cmd = 'python runOfficialTest.py --model=%s --iter=%d' % (mod, iterx)
    subprocess.call(cmd, shell=True)

    # open output file
    eval_fi = osp.join(eval_dir, '%s_%d' % (mod, args['iter']), 'results.txt')
    with open(eval_fi, 'r') as f:
        lines = [line for line in f]
        # make output string
        out += ' %s &' % lines[1][:-1]

    return out



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Remove old / unused models ')
    parser.add_argument('--iter', default=0, type=int, help='minimum number of iterations')
    args = parser.parse_args()
    args = vars(args)
    main(args)

