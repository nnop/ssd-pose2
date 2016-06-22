import argparse
import os 
import os.path as osp
import subprocess
import sys


def main(args):
    #print args

    diff = ""
    if not args['difficult']:
        diff = '--difficult'

    samp = ""
    if args['sampler']:
        samp = '--sampler'

    pose = ""
    if not args['seperate_pose']:
        pose = '--share_pose'
    
    os.chdir('data/3Dpascal/pascal3D')
    pyAnnsCmd = 'python makePyAnns.py --num_bins=%d --num_pascal=%d %s' % \
    (args['num_bins'], args['num_pascal'], diff)
    print pyAnnsCmd
    subprocess.call(pyAnnsCmd, shell=True)
    #print sys.exit()


    os.chdir('../../..')

    data_root_dir='data/3Dpascal/pascal3D'
    mapfile = 'data/3Dpascal/pascal3D/labelmap_3D.prototxt'
    anno_type='detection'
    label_type='json'
    #db='lmdb'
    
    splits = []
    valdb = ('val_lmdb_bins_%d_%d' % (args['num_bins'], args['size']), 'val_bins=%d' % (args['num_bins']), 'val.txt')
    testdb = ('test_lmdb_bins_%d_%d' % (args['num_bins'], args['size']), 'test_bins=%d' % (args['num_bins']), 'test.txt')

    trstem ='train_bins=%d_diff=%r_numPascal=%d' % (args['num_bins'], args['difficult'], args['num_pascal']) 
    trdb = ('%s_lmdb_%d' % (trstem, args['size']), trstem, 'train.txt')
    
    splits.append(valdb)
    splits.append(testdb)
    splits.append(trdb)

    for split in splits:
        listFile = osp.join(data_root_dir, 'cache', split[1], split[2])
        outFile = osp.join(data_root_dir, 'lmdb', split[0])

        cmd = 'python scripts/create_annoset.py --anno-type=%s --label-type=%s \
        --label-map-file=%s --encode-type=jpg --root=%s \
        --listfile=%s --outdir=%s' % \
        (anno_type, label_type, mapfile, data_root_dir, listFile, outFile)

        print cmd
        subprocess.call(cmd, shell=True)
        
    idx = 'bins=%d_diff=%r_numPascal=%d_size=%d_lr=%f_samp=%r' % (args['num_bins'], args['difficult'], \
        args['num_pascal'], args['size'], args['base_lr'], args['sampler'])


    ssdCmd = 'python examples/ssd/ssd_pascal3D.py --train_lmdb=%s --val_lmdb=%s --test_lmdb=%s --idx=%s \
    --gpu1=%d --gpu2=%d --num_bins=%d %s --size=%d --max_iter=%d --base_lr=%f --resume=%r --remove=%r %s' % \
    (osp.join(data_root_dir, 'lmdb', trdb[0]), osp.join(data_root_dir, 'lmdb', valdb[0]), osp.join(data_root_dir, 'lmdb', testdb[0]), idx, args['gpu1'], args['gpu2'], \
        args['num_bins'], pose, args['size'], args['max_iter'], args['base_lr'], args['resume'], \
         args['remove'], samp)
    print ssdCmd
    subprocess.call(ssdCmd, shell=True)







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver for SSD w/ Pose experiments")
    parser.add_argument('--num_bins', default='8', type=int, help='number of bins to divide angles into')
    parser.add_argument('--num_pascal', default='1', type=int, help='number of times to include pascal ims')
    parser.add_argument('--difficult', action='store_false', help='include difficult examples in training')
    parser.add_argument('--sampler', action='store_true', help='include full sampler')
    parser.add_argument('--seperate_pose', action='store_true', help='share pose = class agnostic pose estimation')

    parser.add_argument('--gpu1', default=0, type=int, help='which gpu to train on')
    parser.add_argument('--gpu2', default=-1, type=int, help='which gpu to train on')
    parser.add_argument('--size', default=300, type=int, help='height and width of images')
    parser.add_argument('--max_iter', default=60000, type=int, help='maximum number of iterations')
    parser.add_argument('--base_lr', default=0.00004, type=float, help='base learning rate')
    parser.add_argument('--resume', default=True, type=bool, help='resume training')
    parser.add_argument('--remove', default=False, type=bool, help='remove old models')

    args = parser.parse_args()
    params = vars(args)
    main(params)
