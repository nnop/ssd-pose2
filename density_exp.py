
import os
import sys
sys.path.insert(0, 'python')
import caffe
caffe.set_mode_gpu()
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import json
import os.path as osp
import scipy.io as sio
import argparse
import numpy as np
from utils import options



def main(args):
  
  caffe.set_device(args['gpu'])

  mod_idx = args['idx']

  iterx = args['iter']
  opt = options.Options(osp.join('/home/poirson/options/', mod_idx + '.json'))

  modelW = 'VGG_GMU_%s_iter_%d.caffemodel' % (mod_idx, iterx)
  model_def = osp.join('models/VGGNet/gmu_kitchen/', mod_idx, 'deploy.prototxt')
  model_weights = osp.join('models/VGGNet/gmu_kitchen/', mod_idx, modelW)

  net = caffe.Net(model_def, model_weights, caffe.TEST)     
  transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
  transformer.set_transpose('data', (2, 0, 1))
  transformer.set_mean('data', np.array([104,117,123])) # mean pixel
  transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
  transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

  # phil data
  val = os.listdir('data/scene_one/density/images/')
  if args['test'] != 0:
    f = open('data/scene_one/cache/test_scene=2/test.txt', 'r')
    val = [line.split(' ')[0] for line in f]
    f.close()

  image_resize = opt.get_opts('size')


  all_out = {}
  # hack 
  out_dir = osp.join('data/scene_one/density/', 'det_%s_iter_%d' % (mod_idx, iterx))
  if args['test'] != 0:
    out_dir = osp.join('data/scene_one/density/', 'test_det_%s_iter_%d' % (mod_idx, iterx))

  if not osp.exists(out_dir):
    os.mkdir(out_dir)

  for idx, v in enumerate(val):
      if args['test'] == 0:
        imfile = osp.join('data/scene_one/density/images/', v)
      else:
        imfile = v
        if 'data/scene_one/images/Bedroom_01_1/' in imfile:
          out_f = imfile[len('data/scene_one/images/Bedroom_01_1/'):]
        else:
          out_f = imfile[len('data/scene_one/images/Kitchen_Living_02_1/'):]

      # set net to batch size of 1
      net.blobs['data'].reshape(1,3,image_resize,image_resize)
      
      # hack ??
      if not osp.exists(imfile):
          print imfile
          return
      
      image = caffe.io.load_image(imfile)

      transformed_image = transformer.preprocess('data', image)
      net.blobs['data'].data[...] = transformed_image

      # Forward pass.
      detections = net.forward()['detection_out']

      # Parse the outputs.
      out = np.zeros((detections.shape[2], 6))
      out[:, 0] = detections[0,0,:,3]*image.shape[1]
      out[:, 1] = detections[0,0,:,4]*image.shape[0]
      out[:, 2] = detections[0,0,:,5]*image.shape[1]
      out[:, 3] = detections[0,0,:,6]*image.shape[0]
      out[:, 4] = detections[0,0,:,2]
      out[:, 5] = detections[0,0,:,1]


      fiOutput = out_f[:-4] + '.mat'
      out_file = osp.join(out_dir, fiOutput)
      sio.savemat(out_file, {'dets': out})

      if idx % 100 == 0:
        print '%d/%d' % (idx, len(val))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test an SSD model ")
    parser.add_argument('--idx', type=str, help='model name')
    parser.add_argument('--iter', type=int, help='iteration to test')
    parser.add_argument('--gpu', type=int, default=0, help='gpu to use')
    parser.add_argument('--test', type=int, default=0, help='run on the test set')
    
    args = parser.parse_args()
    params = vars(args)
    main(params)