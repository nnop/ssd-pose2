import argparse
import glob
import json
import os
import os.path as osp
import shutil
import subprocess

from utils import options

'''
Makes the tables for different model evaluations 
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

        # try and get model idx
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

            # get mod to options dictionary
            opt_file = osp.join('/home/poirson/options/', '%d.json' % idx)
            if osp.exists(opt_file):
                opt = options.Options(opt_file)
                mod_to_opts[mod] = opt


    # list to save time
    # if mat_eval called for file
    # append to this list 
    eval_done = []

    eval_dir = 'eval_%d' % args['iter']
    os.mkdir(eval_dir)

    # make share and seperate table
    out_fi = open(osp.join(eval_dir, 'share_sep_eval.txt'), 'w')
    for des in [True, False]:
        # make output string
        out = ''
        if des:
            out += 'Share 300 &'
        else:
            out += 'Separate 300 &'

        for bins in [4, 8, 16, 24]:
            found_mod = False
            for mod, val in mod_to_opts.iteritems():
                if val.get_opts('share_pose') == des and\
                val.get_opts('num_bins') == bins and val.get_opts('rotate')\
                and val.get_opts('imagenet') and val.get_opts('size') == 300:
                    out, eval_done = make_out(mod, args['iter'], out, eval_done)
                    found_mod = True
            if not found_mod:
                out += ' none &'



        # write output
        out_fi.write(out + '\n')
    out_fi.close()


    # make with and without pascal table
    out_fi = open(osp.join(eval_dir, 'no_pascal_eval.txt'), 'w')
    for des in [True, False]:
        out = ''
        if des:
            out += 'Share 300 &'
        else:
            out += 'Share 300 Pascal &'

        for bins in [4, 8, 16, 24]:
            found_mod = False
            for mod, val in mod_to_opts.iteritems():
                if val.get_opts('share_pose') and val.get_opts('num_bins') == bins\
                and val.get_opts('rotate') and val.get_opts('imagenet') == des\
                and val.get_opts('size') == 300:
                    out, eval_done = make_out(mod, args['iter'], out, eval_done)
                    found_mod = True

            if not found_mod:
                out += ' none &'

        out_fi.write(out + '\n')
    out_fi.close()


    # make rotated table
    out_fi = open(osp.join(eval_dir,'rot_eval.txt'), 'w')
    for sz in [300, 500]:
        for rot in [False, True]:
            out = ''
            if rot:
                out += 'Share %d Rot &' % sz
            else:
                out += 'Share %d &' % sz

            for bins in [4, 8, 16, 24]:
                found_mod = False
                for mod, val in mod_to_opts.iteritems():
                    if val.get_opts('share_pose') and val.get_opts('num_bins') == bins\
                    and val.get_opts('rotate') == rot and val.get_opts('imagenet')\
                    and val.get_opts('size') == sz:
                        out, eval_done = make_out(mod, args['iter'], out, eval_done)
                        found_mod = True

                if not found_mod:
                    out += ' none &'

            out_fi.write(out + '\n')
    out_fi.close()

    
    # make large results table
    out_fi = open(osp.join(eval_dir,'all_res.txt'), 'w')
    for bins in [4, 8, 16, 24]:
        out_fi.write('%d bins \n' % bins)
        for sz in [300, 500]:
            out = 'Ours Share %d &' % sz

            for mod, val in mod_to_opts.iteritems():
                if val.get_opts('share_pose') and val.get_opts('num_bins') == bins\
                and val.get_opts('rotate') and val.get_opts('imagenet')\
                and val.get_opts('size') == sz:
                    out, eval_done = get_row(mod, args['iter'], out, eval_done)

            out_fi.write(out + '\n')
    out_fi.close()


def get_row(mod, iterx, out, eval_done):
    if mod not in eval_done:
        cmd = 'python runOfficialTest.py --model=%s --iter=%d' % (mod, iterx)
        subprocess.call(cmd, shell=True)
        eval_done.append(mod)

    # open output file
    eval_fi = osp.join(eval_dir, '%s_%d' % (mod, args['iter']), 'results.txt')
    with open(eval_fi, 'r') as f:
        lines = [line for line in f]
        # make output string
        out += ' %s' % lines[0][:-1]

    return out, eval_done   


def make_out(mod, iterx, out, eval_done):
    if mod not in eval_done:
        cmd = 'python runOfficialTest.py --model=%s --iter=%d' % (mod, iterx)
        subprocess.call(cmd, shell=True)
        eval_done.append(mod)

    # open output file
    eval_fi = osp.join(eval_dir, '%s_%d' % (mod, args['iter']), 'results.txt')
    with open(eval_fi, 'r') as f:
        lines = [line for line in f]
        # make output string
        out += ' %s &' % lines[1][:-1]

    return out, eval_done



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Remove old / unused models ')
    parser.add_argument('--iter', default=0, type=int, help='minimum number of iterations')
    args = parser.parse_args()
    args = vars(args)
    main(args)

