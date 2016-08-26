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
  voc_labelmap_file = 'data/scenes/labelmap.prototxt'
  file = open(voc_labelmap_file, 'r')
  voc_labelmap = caffe_pb2.LabelMap()
  text_format.Merge(str(file.read()), voc_labelmap)

  mod_fi = args['model']
  if 'SSD_share_pose_' in mod_fi:
    opts = mod_fi[len('SSD_share_pose_'):] + '.json'
  elif 'SSD_seperate_pose_' in mod_fi:
    opts = mod_fi[len('SSD_seperate_pose_'):] + '.json'
  else:
    print 'whoops'
    return


  iterx = args['iter']
  opt = options.Options(osp.join('/home/poirson/options/', opts))

  if osp.exists(osp.join('models/VGGNet/scenes/', mod_fi,\
   'VGG_scenes_%s_iter_%d.caffemodel' % (mod_fi, iterx))):
    modelW = 'VGG_scenes_%s_iter_%d.caffemodel' % (mod_fi, iterx)
  elif osp.exists(osp.join('models/VGGNet/scenes/', mod_fi,\
    'VGG_Rohit_%s_iter_%d.caffemodel' % (mod_fi, iterx))):
    modelW = 'VGG_Rohit_%s_iter_%d.caffemodel' % (mod_fi, iterx)
  elif osp.exists(osp.join('models/VGGNet/scenes/', mod_fi,\
    'VGG_Pascal3D_%s_iter_%d.caffemodel' % (mod_fi, iterx))):
    modelW = 'VGG_Pascal3D_%s_iter_%d.caffemodel' % (mod_fi, iterx)
  else:
    print 'whoops'
    return

  model_def = osp.join('models/VGGNet/scenes/', mod_fi, 'deploy.prototxt')
  model_weights = osp.join('models/VGGNet/scenes/', mod_fi, modelW)

  net = caffe.Net(model_def, model_weights, caffe.TEST)     
  transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
  transformer.set_transpose('data', (2, 0, 1))
  transformer.set_mean('data', np.array([104,117,123])) # mean pixel
  transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
  transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

  db_idx = opt.get_scene_db_stem('test')
  f = open(osp.join('data/scenes/', 'cache', db_idx, 'test.txt'), 'r')
  val = [line.strip('\n').split(' ')[0] for line in f ]

  
  #size_loc = mod_fi.find('size=')
  #size_loc += 5
  #image_resize = int(mod_fi[size_loc:size_loc+3])
  image_resize = opt.get_opts('size')

  #bin_loc = mod_fi.find('bins=')
  #bin_loc += 5
  #num_bins = int(mod_fi[bin_loc])
  num_bins = opt.get_opts('num_bins')

  #rotate = False
  #if mod_fi.find('rotate=True') != -1:
  #  rotate = True
  rotate = opt.get_opts('rotate')


  all_out = {}
  # hack 
  fiOutput = 'scenes_%s_%d' % (mod_fi, iterx)
  skip = False
  if osp.exists(osp.join('mat_eval', fiOutput, 'chair.mat')):
    skip = True


  for idx, v in enumerate(val):
      if skip:
        break
      imfile = v

      # set net to batch size of 1
      net.blobs['data'].reshape(1,3,image_resize,image_resize)
      
      # hack ??
      #if not osp.exists(imfile):
      #    imfile = osp.join('data/scenes/PASCAL/VOCdevkit/VOC2012/JPEGImages/', v + '.jpg')
          #print 'why'
      
      image = caffe.io.load_image(imfile)

      #plt.imshow(image)


      transformed_image = transformer.preprocess('data', image)
      net.blobs['data'].data[...] = transformed_image

      # Forward pass.
      detections = net.forward()['detection_out']

      # Parse the outputs.
      det_label = get_labelname(voc_labelmap, detections[0,0,:,1].tolist())
      out = np.zeros((detections.shape[2], 6))
      out[:, 0] = detections[0,0,:,3]*image.shape[1]
      out[:, 1] = detections[0,0,:,4]*image.shape[0]
      out[:, 2] = detections[0,0,:,5]*image.shape[1]
      out[:, 3] = detections[0,0,:,6]*image.shape[0]
      out[:, 4] = detections[0,0,:,7]
      out[:, 5] = detections[0,0,:,2]
          
      cur_out = {'diningtable':[],
                'sofa':[],
                'monitor':[],
                'chair':[]}
      
      for det_idx, lbl in enumerate(det_label):
          cur_out[lbl].append(out[det_idx,:])
      
      for lbl, v in cur_out.iteritems():
          if lbl not in all_out:
              all_out[lbl] = []
          all_out[lbl].append(v)
      
      if idx % 100 == 0:
          print '%d/%d' % (idx, len(val)) 

  for lbl, all_val in all_out.iteritems():
      if skip:
        break
      if not osp.exists('scene_mat_eval'):
        os.mkdir('scene_mat_eval')
      if not osp.exists(osp.join('scene_mat_eval', fiOutput)):
        os.mkdir(osp.join('scene_mat_eval', fiOutput))

      out_file = osp.join('scene_mat_eval', fiOutput, lbl + '.mat')
      sio.savemat(out_file, {'dets': all_val})

  #extra = ' rotate = false; '
  #if rotate:
  #  extra = ' rotate = true; '

  #if args['test_bins'] != -1:
  #  num_bins = args['test_bins']
  
  # make test scene gts
  # hack
  prepare_test_gt(db_idx)

  # hard coded hack
  # TODO use options instead
  sc_path = db_idx[-7:]
  print 
    
  matlab_cmd = 'bins = %d; scene_path=\'%s\'; path = \'%s\'; scene_avp_eval;' % (num_bins, sc_path, osp.join('scene_mat_eval', fiOutput))
  print matlab_cmd
  os.system('matlab -nodisplay -r "try %s catch; end; quit;"' % (matlab_cmd))


def prepare_test_gt(fold):
  data_path = 'data/scenes/cache/'

  out_path = 'data/scenes/matTest/'

  if not osp.exists(out_path):
    os.mkdir(out_path)

  out_dir = osp.join(out_path, fold[-7:])

  if not osp.exists(out_dir):
    os.mkdir(out_dir)

  f = open(osp.join(data_path, fold, 'test.txt'), 'r')

  jsons = [line.strip('\n').split(' ')[1][len(osp.join(data_path, fold))+1:] for line in f]
  f.close()

  out_f = open(osp.join(out_dir, 'jsons.txt'), 'w')

  for js in jsons:
    out_f.write(js[:-5] + '\n')
    d_fi = osp.join(data_path, fold, js)
    data = json.load(open(d_fi, 'r'))
    
    rec = {}
    objs = []
    for idx, ann in enumerate(data['annotation']):
        temp = {}
        temp['class'] = ann['category_id']
        temp['azimuth'] = ann['viewpoint']['azimuth']
        temp['bbox'] = ann['bbox']
        objs.append(temp)
    rec['objects'] = np.asarray(objs)
    sio.savemat(osp.join(out_dir, js[:-5] + '.mat'), rec)


def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test an SSD model ")
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--iter', type=int, help='iteration to test')
    parser.add_argument('--test_bins', type=int, default=-1, help='bins to test with 24 bin model')
    parser.add_argument('--gpu', type=int, default=0, help='gpu to use')
    
    args = parser.parse_args()
    params = vars(args)
    main(params)