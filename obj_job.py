import argparse
import os 
import os.path as osp
import subprocess
import sys
import random

from utils import options
from examples.ssd import ssd_ObjNet



def main(args):
    opt_dir = args['opt']

    if args['idx'] != 0 and opt_dir == '':
        opt_dir = osp.join('/home/poirson/options/', str(args['idx']) + '.json')
    
    opt = options.Options(opt_dir)

    mod_id = random.randint(1, 999000)
    if args['idx'] != 0:
        mod_id = args['idx']

    opt.add_kv('mod_id', mod_id)

    opt_out_path = osp.join('/home/poirson/options', '%d.json' % mod_id)
    opt.write_opt(opt_out_path)


    ssd = ssd_ObjNet.O3DSSD()
    ssd.run_main(opt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Driver for SSD w/ Pose experiments')
    parser.add_argument('--opt', default='', type=str, help='path to json file with options')
    parser.add_argument('--idx', default=0, type=int, help='specify model id to resume')


    args = parser.parse_args()
    params = vars(args)
    main(params)

