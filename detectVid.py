import numpy as np 
import os
import sys
import argparse
#import timeit
#import json
import matplotlib.pyplot as plt
import os.path as osp
sys.path.insert(0, 'python')

import caffe
caffe.set_mode_gpu()
caffe.set_device(0)
from google.protobuf import text_format
from caffe.proto import caffe_pb2


#scene = 'video_sequence_3'


def main(args):
	voc_labelmap_file = 'data/philData/labelmap.prototxt'
	file = open(voc_labelmap_file, 'r')
	voc_labelmap = caffe_pb2.LabelMap()
	text_format.Merge(str(file.read()), voc_labelmap)


	model_def = '/playpen/poirson/ssd-pose/models/VGGNet/philData/SSD_share_pose_bins=8_size=300_lr=0.000040_samp=False_weight=1.250000_step=6000_rotate=False/deploy.prototxt'
	model_weights = '/playpen/poirson/ssd-pose/models/VGGNet/philData/SSD_share_pose_bins=8_size=300_lr=0.000040_samp=False_weight=1.250000_step=6000_rotate=False/VGG_Pascal3D_SSD_share_pose_bins=8_size=300_lr=0.000040_samp=False_weight=1.250000_step=6000_rotate=False_iter_24000.caffemodel'

	net = caffe.Net(model_def, model_weights, caffe.TEST)

	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2, 0, 1))
	transformer.set_mean('data', np.array([104,117,123])) # mean pixel
	transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
	transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

	scene = 'video_sequence_%d' % args['vid_seq_id']

	basePath = 'data/vid/'
	outDir = osp.join(basePath,'vidOutput', scene)

	if not osp.exists(outDir):
		os.makedirs(outDir)
	
	seq = os.listdir(osp.join(basePath, scene))
	seq = sorted(seq)

	for idx, fname in enumerate(seq):
		if idx % 20 == 0:
			print '%d/%d processed' % (idx, len(seq))
		plt.figure(figsize=(20,15))

		imfile = osp.join(basePath, scene, seq[idx])
		image_resize = 300
		net.blobs['data'].reshape(1,3,image_resize,image_resize)

		image = caffe.io.load_image(imfile)
		plt.imshow(image)

		transformed_image = transformer.preprocess('data', image)
		net.blobs['data'].data[...] = transformed_image

		#print 'det start'
		#def det_fun():
		#   detections = net.forward()['detection_out']
		
		#print timeit.timeit(det_fun, number=10) / 10.0

		detections = net.forward()['detection_out']

		# Parse the outputs.
		det_label = detections[0,0,:,1]
		det_conf = detections[0,0,:,2]
		det_xmin = detections[0,0,:,3]
		det_ymin = detections[0,0,:,4]
		det_xmax = detections[0,0,:,5]
		det_ymax = detections[0,0,:,6]
		det_pose = detections[0, 0, :, 7]
		det_poseScore = detections[0,0,:,8]


		# Get detections with confidence higher than 0.6.
		top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.4]

		top_conf = det_conf[top_indices]
		top_label_indices = det_label[top_indices].tolist()
		#print top_label_indices
		top_labels = get_labelname(voc_labelmap, top_label_indices)
		top_xmin = det_xmin[top_indices]
		top_ymin = det_ymin[top_indices]
		top_xmax = det_xmax[top_indices]
		top_ymax = det_ymax[top_indices]
		top_pose = det_pose[top_indices]
		top_poseScore = det_poseScore[top_indices]

		#print det_conf
		#print det_conf.shape
		#print det_ymax.shape
		#print det_pose.shape

		colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

		plt.imshow(image)
		currentAxis = plt.gca()

		for i in xrange(top_conf.shape[0]):
			xmin = int(round(top_xmin[i] * image.shape[1]))
			ymin = int(round(top_ymin[i] * image.shape[0]))
			xmax = int(round(top_xmax[i] * image.shape[1]))
			ymax = int(round(top_ymax[i] * image.shape[0]))
			score = top_conf[i]
			label = top_labels[i]
			pose = top_pose[i]
			pose_score = top_poseScore[i]
			name = '%s: %.2f, predicted pose = %d(%.2f)'%(label, score, int(pose), pose_score)
			coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
			color = colors[i % len(colors)]
			currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
			currentAxis.text(xmin, ymin, name, bbox={'facecolor':'white', 'alpha':0.4})

		plt.tight_layout()
		plt.savefig(osp.join(outDir, fname))
		plt.close()

	print 'done'





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
	parser = argparse.ArgumentParser()
	parser.add_argument('--vid_seq_id', default=1, type=int, help='int id of vid sequence')

	args = parser.parse_args()
	params = vars(args)
	main(params)