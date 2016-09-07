import argparse
import os 
import os.path as osp
import subprocess
import sys
import random

from utils import options
from utils import makeRohitAnns
from examples.ssd import ssd_GMU

DATA_DIR = 'data/gmu_kitchen/'

def main(args):

    data_root_dir = DATA_DIR

    opt_dir = args['opt']

    if args['idx'] != 0 and opt_dir == '':
        opt_dir = osp.join('/home/poirson/options/', str(args['idx']) + '.json')

    opt = options.Options(opt_dir)
    anns = makeRohitAnns.MakeAnns(opt)
    anns.run_main()

    mapfile = osp.join(DATA_DIR, 'labelmap.prototxt')
    anno_type='detection'
    label_type='json'


    trstem = opt.get_gmu_db_stem('train') 
    trdb = ('%s_lmdb' % trstem, trstem, 'train.txt')

    valstem = opt.get_gmu_db_stem('val') 
    valdb = ('%s_lmdb' % valstem, valstem, 'val.txt')

    testem = opt.get_gmu_db_stem('test') 
    testdb = ('%s_lmdb' % testem, testem, 'test.txt')


    with open(osp.join(data_root_dir, 'cache', testem, 'test.txt'), 'r') as infile:
        numTest = len([line for line in infile]) 
        opt.add_kv('num_test', numTest)

    with open(osp.join(data_root_dir, 'cache', valstem, 'val.txt'), 'r') as infile:
        numTest = len([line for line in infile]) 
        opt.add_kv('num_val', numTest)

    splits = [trdb, valdb, testdb]
    

    for split in splits:
        listFile = osp.join(DATA_DIR, 'cache', split[1], split[2])
        outFile = osp.join(DATA_DIR, 'lmdb', split[0])

        cmd = 'python scripts/create_annoset.py --anno-type=%s --label-type=%s \
        --label-map-file=%s --encode-type=jpg --root=%s \
        --listfile=%s --outdir=%s' % \
        (anno_type, label_type, mapfile, './', listFile, outFile)

        print cmd
        subprocess.call(cmd, shell=True)

    mod_id = random.randint(1, 999000)
    if args['idx'] != 0:
        mod_id = args['idx']

    opt.add_kv('mod_id', mod_id)

    opt_out_path = osp.join('/home/poirson/options', '%d.json' % mod_id)
    opt.write_opt(opt_out_path)

    ssd = ssd_GMU.SSD()
    ssd.run_main(opt)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver for SSD w/ Pose experiments")

    parser.add_argument('--opt', default='', type=str, help='path to json file with options')
    parser.add_argument('--idx', default=0, type=int, help='specify model id to resume')

    args = parser.parse_args()
    params = vars(args)
    main(params)
