import argparse
import os 
import os.path as osp
import subprocess
import sys
import random

from utils import options
from utils import makeP3DAnns
from examples.ssd import ssd_pascal3D



def main(args):

    opt_path = args['opt']
    if args['idx'] != 0:
        opt_path = osp.join('/home/poirson/options', '%d.json' % args['idx'])

    opt = options.Options(opt_path)
    anns = makeP3DAnns.MakeAnns(opt)
    anns.run_main()

    data_root_dir='data/pascal3D'
    #data_root_dir = ''
    mapfile = 'data/pascal3D/labelmap_3D.prototxt'
    anno_type='detection'
    label_type='json'


    #trstem ='train_bins=%d_diff=%r_imgnet=%r_numPascal=%d_rotate=%r' \
    #% (args['num_bins'], args['difficult'], args['imagenet'], args['num_pascal'], args['rotate'])
    trstem = opt.get_db_name_stem('train') 
    trdb = ('%s_lmdb' % trstem, trstem, 'train.txt')

    valstem = opt.get_db_name_stem('val') 
    valdb = ('%s_lmdb' % valstem, valstem, 'val.txt')

    testem = opt.get_db_name_stem('test') 
    testdb = ('%s_lmdb' % testem, testem, 'test.txt')
    

    with open(osp.join(data_root_dir, 'cache', valstem, 'val.txt'), 'r') as infile:
        numVal = len([line for line in infile])
        opt.add_kv('num_val', numVal)

    with open(osp.join(data_root_dir, 'cache', testem, 'test.txt'), 'r') as infile:
        numTest = len([line for line in infile]) 
        opt.add_kv('num_test', numTest)
    
    splits = []
    splits.append(valdb)
    splits.append(testdb)
    splits.append(trdb)
    #print splits
    
    # still hacky 
    for split in splits:
        listFile = osp.join(data_root_dir, 'cache', split[1], split[2])
        outFile = osp.join(data_root_dir, 'lmdb', split[0])

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

    ssd = ssd_pascal3D.P3DSSD()
    ssd.run_main(opt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Driver for SSD w/ Pose experiments')
    parser.add_argument('--opt', default='', type=str, help='path to json file with options')
    parser.add_argument('--idx', default=0, type=int, help='specify model id to resume')


    args = parser.parse_args()
    params = vars(args)
    main(params)
