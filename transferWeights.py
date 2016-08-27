from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import argparse


# Make sure that caffe is on the python path:
caffe_root = './'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import os

import caffe
caffe.set_mode_cpu()


from google.protobuf import text_format
from caffe.proto import caffe_pb2


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


def sample_param(src_param, src_num_classes, dst_num_classes, num_bboxes, maps):
	src_shape = src_param.shape
	assert src_shape[0] == src_num_classes * num_bboxes
	if len(src_shape) == 4:
		dst_shape = (dst_num_classes * num_bboxes, src_shape[1], src_shape[2], src_shape[3])
	else:
		dst_shape = dst_num_classes * num_bboxes
	dst_param = np.zeros(dst_shape)
	for i in xrange(0, num_bboxes):
		for m in maps:
			[src_label, dst_label, name] = m
			src_idx = i * src_num_classes + int(src_label)
			dst_idx = i * dst_num_classes + int(dst_label)
			dst_param[dst_idx,] = src_param[src_idx,]
	return dst_param



def main(args):


	resolution = 300
	p3d_dir = args['p3d_model']
	p3d_model = 'VGG_Pascal3D_%s_iter_%d.caffemodel' % (args['p3d_model'], args['p3d_iter'])

	rohit_dir = args['rohit_model']

	num_poses = args['num_poses']

	pas_deploy = osp.join('models/VGGNet/Pascal3D/', p3d_dir, 'deploy.prototxt')
	phil_deploy = osp.join('models/VGGNet/scenes/', rohit_dir,'deploy.prototxt')

	model_name = 'VGG_scenes_%s' % (rohit_dir)

	max_iter = 0
	# Find most recent snapshot.
	for file in os.listdir(osp.join('models/VGGNet/scenes/', rohit_dir)):
	  if file.endswith(".solverstate"):
		basename = os.path.splitext(file)[0]
		#print(basename)
		#print()
		#print(basename.split("{}_iter_".format(model_name)))
		iter = int(basename.split("{}_iter_".format(model_name))[1])
		if iter > max_iter:
		  max_iter = iter

	rohit_model = '%s_iter_%d.caffemodel' % (model_name, max_iter)

 
	new_voc_model_dir = osp.join(caffe_root, 'models/VGGNet/scenes/', rohit_dir)

	pas_net = caffe.Net(caffe_root + pas_deploy, osp.join(caffe_root, 'models/VGGNet/Pascal3D/', p3d_dir , p3d_model), caffe.TEST)

	#phil_net = caffe.Net(caffe_root + phil_deploy, osp.join(caffe_root, 'models/VGGNet/scenes/', rohit_dir, rohit_model), caffe.TEST)
	#phil_net = caffe.Net(caffe_root + phil_deploy, caffe.TEST)


	# load MS COCO model specs
	file = open(caffe_root + pas_deploy, 'r')
	pas_netspec = caffe_pb2.NetParameter()
	text_format.Merge(str(file.read()), pas_netspec)

	# load MS COCO labels
	pas_labelmap_file = caffe_root + 'data/3Dpascal/pascal3D/labelmap_3D.prototxt'
	file = open(pas_labelmap_file, 'r')
	pas_labelmap = caffe_pb2.LabelMap()
	text_format.Merge(str(file.read()), pas_labelmap)

	# load PASCAL VOC model specs
	file = open(caffe_root + phil_deploy, 'r')
	voc_netspec = caffe_pb2.NetParameter()
	text_format.Merge(str(file.read()), voc_netspec)

	# load PASCAL VOC labels
	voc_labelmap_file = caffe_root + 'data/scenes/labelmap.prototxt'
	file = open(voc_labelmap_file, 'r')
	voc_labelmap = caffe_pb2.LabelMap()
	text_format.Merge(str(file.read()), voc_labelmap)



	for layer_name, param in pas_net.params.iteritems():
		if len(param) == 2:
			print(layer_name + '\t' + str(param[0].data.shape) + str(param[1].data.shape))
		else:
			print(layer_name + '\t' + str(param[0].data.shape))




	map_file =  'p3d_rohit_map.txt'
	if not os.path.exists(map_file):
		print('{} does not exist'.format(map_file))
	
	maps = np.loadtxt(map_file, str, delimiter=',')

	for m in maps:
		[coco_label, voc_label, name] = m
		coco_name = get_labelname(pas_labelmap, int(coco_label))[0]
		voc_name = get_labelname(voc_labelmap, int(voc_label))[0]
		#print(coco_name)
		#print(voc_name)
		#print(name)
		assert voc_name == name
		print('{}, {}'.format(coco_name, voc_name))



	mbox_source_layers = ['conv4_3_norm', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'pool6']
	num_bboxes = [3, 6, 6, 6, 6, 6]

	assert len(mbox_source_layers) == len(num_bboxes)
	num_voc_classes = 5
	num_coco_classes = 12

	for i in xrange(0, len(mbox_source_layers)):
		mbox_source_layer = mbox_source_layers[i]
		mbox_priorbox_layer = '{}_mbox_priorbox'.format(mbox_source_layer)
		mbox_loc_layer = '{}_mbox_loc'.format(mbox_source_layer)
		mbox_conf_layer = '{}_mbox_conf'.format(mbox_source_layer)
		mbox_pose_layer = '{}_mbox_pose'.format(mbox_source_layer)
		num_bbox = num_bboxes[i]
		for j in xrange(0, len(pas_netspec.layer)):
			layer = pas_netspec.layer[j]
			if mbox_priorbox_layer == layer.name:
				voc_netspec.layer[j].prior_box_param.CopyFrom(layer.prior_box_param)
			if mbox_loc_layer == layer.name:
				voc_netspec.layer[j].convolution_param.num_output = num_bbox * 4
			if mbox_conf_layer == layer.name:
				voc_netspec.layer[j].convolution_param.num_output = num_bbox * num_voc_classes
			if mbox_pose_layer == layer.name:
				voc_netspec.layer[j].convolution_param.num_output = num_bbox * num_poses

	if not os.path.exists(new_voc_model_dir):
		os.makedirs(new_voc_model_dir)
	# del voc_netspec.layer[-1]

	new_voc_model_def_file = '{}/deploy.prototxt'.format(new_voc_model_dir)
	with open(new_voc_model_def_file, 'w') as f:
		print(voc_netspec, file=f)

	voc_net_new = caffe.Net(new_voc_model_def_file, caffe.TEST)

	new_voc_model_file = '{}/{}'.format(new_voc_model_dir, rohit_model)


	for layer_name, param in pas_net.params.iteritems():
		if 'mbox' not in layer_name:
			for i in xrange(0, len(param)):
				voc_net_new.params[layer_name][i].data.flat = pas_net.params[layer_name][i].data.flat

	for i in xrange(0, len(mbox_source_layers)):
		layer = mbox_source_layers[i]
		num_bbox = num_bboxes[i]
		conf_layer = '{}_mbox_conf'.format(layer)
		voc_net_new.params[conf_layer][0].data.flat = sample_param(pas_net.params[conf_layer][0].data,
														  len(pas_labelmap.item), len(voc_labelmap.item), num_bbox, maps)
		voc_net_new.params[conf_layer][1].data.flat = sample_param(pas_net.params[conf_layer][1].data,
														  len(pas_labelmap.item), len(voc_labelmap.item), num_bbox, maps)
		loc_layer = '{}_mbox_loc'.format(layer)
		voc_net_new.params[loc_layer][0].data.flat = pas_net.params[loc_layer][0].data.flat
		voc_net_new.params[loc_layer][1].data.flat = pas_net.params[loc_layer][1].data.flat
		
		pose_layer = '{}_mbox_pose'.format(layer)
		voc_net_new.params[pose_layer][0].data.flat = pas_net.params[pose_layer][0].data.flat
		voc_net_new.params[pose_layer][1].data.flat = pas_net.params[pose_layer][1].data.flat
	
	voc_net_new.save(new_voc_model_file)
	print(new_voc_model_file)
	print(model_name)
	print(max_iter)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--p3d_model', type=str, help='pascal 3d direcotry ')
	parser.add_argument('--rohit_model', type=str, help='rohit model direcotry ')
	parser.add_argument('--p3d_iter', type=int, help='pascal 3d iteration ')
	parser.add_argument('--num_poses', type=int, help='number of poses')
	#parser.add_argument('--rohit_iter', type=int, help='rohit iteration ')

	args = parser.parse_args()
	params = vars(args)
	main(params)

