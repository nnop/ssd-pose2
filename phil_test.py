import os.path as osp
import argparse
import subprocess



# args 
#model to test name
# iteration to load
def main(args):
    base_dir = 'models/VGGNet/philData/'
    file_name = args['model']

    test_solver_file = osp.join(base_dir, file_name, 'test_solver.prototxt')
    snapshot_file = 'VGG_Pascal3D_%s_iter_%d.solverstate' % (file_name, args['iter'])
    snapshot = osp.join(base_dir, file_name, snapshot_file)

    testcmd =   './build/tools/caffe train --solver=%s --snapshot=%s --gpu=%s' % (test_solver_file, snapshot, args['gpu'])
    subprocess.call(testcmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test an SSD model ")
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--iter', type=int, help='iteration to test')
    parser.add_argument('--gpu', type=str, help='which gpus to use')
    
    args = parser.parse_args()
    params = vars(args)
    main(params)
