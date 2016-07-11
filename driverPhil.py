import argparse
import os 
import os.path as osp
import subprocess
import sys


DATA_DIR = 'data/philData/'

def main(args):
    #print args

    samp = ""
    if args['sampler']:
        samp = '--sampler'

    pose = ""
    if not args['seperate_pose']:
        pose = '--share_pose'

    rot = ''
    if args['rotate']:
        rot = '--rotate'
    
    pyAnnsCmd = 'python makeAnnsPhil.py --num_bins=%d --test_scene_id=%d %s' % (args['num_bins'], args['test_scene_id'], rot)
    print pyAnnsCmd
    subprocess.call(pyAnnsCmd, shell=True)
    #print sys.exit()

    mapfile = osp.join(DATA_DIR, 'labelmap.prototxt')
    anno_type='detection'
    label_type='json'
    #db='lmdb'

    trstem ='train_bins=%d_rotate=%r_excl=%d' % (args['num_bins'], args['rotate'], args['test_scene_id']) 
    
    trdb = ('%s_lmdb' % (trstem), trstem, 'train.txt')
    
    #valstem = 'val_bins=%d_rotate=%r' % (args['num_bins'], args['rotate'])
    #valdb = ('%s_lmdb' % (valstem), valstem, 'val.txt')
    
    testem = 'test_bins=%d_rotate=%r_scene=%d' % (args['num_bins'], args['rotate'], args['test_scene_id'])
    testdb = ('%s_lmdb' % (testem), testem, 'test.txt')

    #with open(osp.join(DATA_DIR, 'cache', valstem, 'val.txt'), 'r') as infile:
    #    numVal = len([line for line in infile])

    with open(osp.join(DATA_DIR, 'cache', testem, 'test.txt'), 'r') as infile:
        numTest = len([line for line in infile]) 
    
    splits = []
    #splits.append(valdb)
    splits.append(testdb)
    splits.append(trdb)
    #print splits


    for split in splits:
        listFile = osp.join(DATA_DIR, 'cache', split[1], split[2])
        outFile = osp.join(DATA_DIR, 'lmdb', split[0])

        cmd = 'python scripts/create_annoset.py --anno-type=%s --label-type=%s \
        --label-map-file=%s --encode-type=jpg --root=./ \
        --listfile=%s --outdir=%s' % \
        (anno_type, label_type, mapfile, listFile, outFile)

        print cmd
        subprocess.call(cmd, shell=True)
        
    idx = 'test=%d_bins=%d_size=%d_lr=%f_samp=%r_weight=%f_step=%d_rotate=%r' % (args['test_scene_id'], args['num_bins'], \
        args['size'], args['base_lr'], args['sampler'], args['pose_weight'], args['stepsize'], args['rotate'])


    ssdCmd = 'python examples/ssd/ssd_phil.py --train_lmdb=%s  --test_lmdb=%s --idx=%s \
    --gpu=%s --num_bins=%d %s --size=%d --max_iter=%d --base_lr=%f --resume=%r --remove=%r %s --num_test=%d --pose_weight=%f --stepsize=%d' % \
        (osp.join(DATA_DIR, 'lmdb', trdb[0]), osp.join(DATA_DIR, 'lmdb', testdb[0]), idx, args['gpu'], args['num_bins'], \
        pose, args['size'], args['max_iter'], args['base_lr'], args['resume'], args['remove'], samp, numTest, args['pose_weight'], args['stepsize']) 
    print ssdCmd
    subprocess.call(ssdCmd, shell=True)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver for SSD w/ Pose experiments")
    parser.add_argument('--num_bins', default='8', type=int, help='number of bins to divide angles into')
    parser.add_argument('--num_pascal', default='1', type=int, help='number of times to include pascal ims 7 or 8 recommended')
    parser.add_argument('--sampler', action='store_true', help='include full sampler')
    parser.add_argument('--seperate_pose', action='store_true', help='share pose = class agnostic pose estimation')
    parser.add_argument('--imagenet', action='store_false', help='exclude imagenet images')
    parser.add_argument('--pose_weight', default=1.0, type=float, help='weight the pose')
    parser.add_argument('--stepsize', default=20000, type=int, help='step size ')
    parser.add_argument('--rotate', action='store_true', help='bin angles non standard way')
    parser.add_argument('--test_scene_id', type=int, help='int id of the scene to use for testing')


    #parser.add_argument('--gpu1', default=0, type=int, help='which gpu to train on')
    #parser.add_argument('--gpu2', default=-1, type=int, help='which gpu to train on')
    parser.add_argument('--gpu', default='0', type=str, help='which gpus to use seperated by commas')
    parser.add_argument('--size', default=300, type=int, help='height and width of images')
    parser.add_argument('--max_iter', default=60000, type=int, help='maximum number of iterations')
    parser.add_argument('--base_lr', default=0.00004, type=float, help='base learning rate')
    parser.add_argument('--resume', default=True, type=bool, help='resume training')
    parser.add_argument('--remove', default=False, type=bool, help='remove old models')

    args = parser.parse_args()
    params = vars(args)
    main(params)
