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
    #print args

    '''
    diff = ""
    if not args['difficult']:
        diff = '--difficult'

    imgnet = ''
    if not args['imagenet']:
        imgnet = '--imagenet'

    samp = ""
    if args['sampler']:
        samp = '--sampler'

    pose = ""
    if not args['seperate_pose']:
        pose = '--share_pose'

    rot = ''
    if args['rotate']:
        rot = '--rotate'


    
    os.chdir('data/3Dpascal/pascal3D')
    '''
    '''
    pyAnnsCmd = 'python makePyAnns.py --num_bins=%d --num_pascal=%d %s %s %s' % \
    (args['num_bins'], args['num_pascal'], diff, imgnet, rot)
    print pyAnnsCmd
    subprocess.call(pyAnnsCmd, shell=True)
    #print sys.exit()


    os.chdir('../../..')
    '''



    opt = options.Options(args['opt'])
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
        
    #idx = 'bins=%d_diff=%r_imgnet=%r_numPascal=%d_size=%d_lr=%f_samp=%r_weight=%f_step=%d_rotate=%r' % (args['num_bins'], args['difficult'], \
    #    args['imagenet'], args['num_pascal'], args['size'], args['base_lr'], args['sampler'], args['pose_weight'], args['stepsize'], args['rotate'])

    mod_id = random.randint(1, 999000)
    if args['idx'] != 0:
        mod_id = args['idx']

    opt.add_kv('mod_id', mod_id)

    opt_out_path = osp.join('options', '%d.json' % mod_id)
    opt.write_opt(opt_out_path)

    #sys.exit()

    '''
    ssdCmd = 'python examples/ssd/ssd_pascal3D.py --train_lmdb=%s --val_lmdb=%s --test_lmdb=%s --idx=%s \
    --gpu=%s --num_bins=%d %s --size=%d --max_iter=%d --base_lr=%f --resume=%r --remove=%r %s --num_val=%d --num_test=%d --pose_weight=%f --stepsize=%d' % \
        (osp.join(data_root_dir, 'lmdb', trdb[0]), osp.join(data_root_dir, 'lmdb', valdb[0]), \
        osp.join(data_root_dir, 'lmdb', testdb[0]), idx, args['gpu'], args['num_bins'], \
        pose, args['size'], args['max_iter'], args['base_lr'], args['resume'], args['remove'], samp, numVal, numTest, args['pose_weight'], args['stepsize']) 
    print ssdCmd
    subprocess.call(ssdCmd, shell=True)
    '''
    ssd = ssd_pascal3D.P3DSSD()
    ssd.run_main(opt)





if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser(description="Driver for SSD w/ Pose experiments")
    parser.add_argument('--num_bins', default='8', type=int, help='number of bins to divide angles into')
    parser.add_argument('--num_pascal', default='1', type=int, help='number of times to include pascal ims 7 or 8 recommended')
    parser.add_argument('--difficult', action='store_false', help='filter difficult examples from the dataset')
    parser.add_argument('--sampler', action='store_true', help='include full sampler')
    parser.add_argument('--seperate_pose', action='store_true', help='share pose = class agnostic pose estimation')
    parser.add_argument('--imagenet', action='store_false', help='exclude imagenet images')
    parser.add_argument('--pose_weight', default=1.0, type=float, help='weight the pose')
    parser.add_argument('--stepsize', default=20000, type=int, help='step size ')
    parser.add_argument('--rotate', action='store_true', help='bin angles non standard way')

    #parser.add_argument('--gpu1', default=0, type=int, help='which gpu to train on')
    #parser.add_argument('--gpu2', default=-1, type=int, help='which gpu to train on')
    parser.add_argument('--gpu', default='0', type=str, help='which gpus to use seperated by commas')
    parser.add_argument('--size', default=300, type=int, help='height and width of images')
    parser.add_argument('--max_iter', default=60000, type=int, help='maximum number of iterations')
    parser.add_argument('--base_lr', default=0.00004, type=float, help='base learning rate')
    parser.add_argument('--resume', default=True, type=bool, help='resume training')
    parser.add_argument('--remove', default=False, type=bool, help='remove old models')
    '''
    parser = argparse.ArgumentParser(description='Driver for SSD w/ Pose experiments')
    parser.add_argument('--opt', default='', type=str, help='path to json file with options')
    parser.add_argument('--idx', default=0, type=int, help='specify model id to resume')


    args = parser.parse_args()
    params = vars(args)
    main(params)
