import matplotlib
matplotlib.use('Agg')
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

	mod_base = 'models/VGGNet/philData/'
	mod_idx = args['model']
	iterx = args['iter']

	model_def = osp.join(mod_base, mod_idx, 'deploy.prototxt')
	#model_weights = 'VGG_Rohit_%s_iter_%d.caffemodel' % (mod_idx, iterx)
	model_weights = 'VGG_Pascal3D_%s_iter_%d.caffemodel' % (mod_idx, iterx)
	mod_file = osp.join(mod_base, mod_idx, model_weights)

	net = caffe.Net(model_def, mod_file, caffe.TEST)

	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2, 0, 1))
	transformer.set_mean('data', np.array([104,117,123])) # mean pixel
	transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
	transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

	scene_out = 'video_sequence_%d_%s' % (args['vid_seq_id'], mod_idx)
	scene = 'video_sequence_%d' % args['vid_seq_id']

	basePath = 'data/vid/'
	outDir = osp.join(basePath,'vidOutput', scene_out)

	if not osp.exists(outDir):
		os.makedirs(outDir)
	
	seq = os.listdir(osp.join(basePath, scene))
	seq = sorted(seq)

	#SMOTHING CODE
	XMIN = 0
	YMIN = 1
	XMAX = 2
	YMAX = 3
	DET_SCORE = 4
	POSE = 5
	POSE_SCORE = 6
	LABEL = 7
	MATCHING_BOX_ID = 8 

	
	prev_boxes = np.empty((0,9)) 
	#END SMOTHING CODE






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

		keep_box = [0]*len(xrange(top_conf.shape[0]))

		for i in xrange(top_conf.shape[0]):
			xmin = int(round(top_xmin[i] * image.shape[1]))
			ymin = int(round(top_ymin[i] * image.shape[0]))
			xmax = int(round(top_xmax[i] * image.shape[1]))
			ymax = int(round(top_ymax[i] * image.shape[0]))
			score = top_conf[i]
			label = top_labels[i]
			pose = top_pose[i]
			pose_score = top_poseScore[i]





			#SMOTHING CODE
			prev_box = []
			prev_box_ind = -1
			max_iou = 0;
			for jl in range(prev_boxes.shape[0]):
				pos_box = prev_boxes[jl]
				if(pos_box[LABEL] != label):
					continue
				cur_iou = get_bboxes_iou([xmin,ymin,xmax,ymax],
							 pos_box[XMIN:(YMAX+1)])

				if(cur_iou > max_iou):
					max_iou = cur_iou
					prev_box = pos_box
					prev_box_ind = jl

			#end for jl

			#if this box has a previous matching box 
			if(max_iou > .4):
				#if this is the first box in this frame to match with this box	
				if(int(float(prev_box[MATCHING_BOX_ID])) == -1):
					prev_box[MATCHING_BOX_ID] = i
					prev_boxes[prev_box_ind] = prev_box
				else:
					omi = int(float(prev_box[MATCHING_BOX_ID]))
					#if this box has a higher score than the previous matching box
					if(top_conf[omi] < score):
						prev_box[MATCHING_BOX_ID] = i
						prev_boxes[prev_box_ind] = prev_box
						keep_box[i] = 1
						keep_box[omi] = 0
				#end if matching box id

				if(int(float(prev_box[MATCHING_BOX_ID])) == i):
					if(abs(pose - float(prev_box[POSE])) > 1):
						top_pose[i] = float(prev_box[POSE])
			else:
				prev_box = []
				if(score > .7):
					keep_box[i] = 1 

			#end if max_iou > .5




			#END SMOTHING CODE





		#	name = '%s: %.2f, predicted pose = %d(%.2f)'%(label, score, int(pose), pose_score)
		#	coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
		#	color = colors[i % len(colors)]
		#	currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
		#	currentAxis.text(xmin, ymin, name, bbox={'facecolor':'white', 'alpha':0.4})


		#end for i, each box found in frame



		#SMOTHING CODE

		#now see if there are any prev boxes that should be kept
		kept_boxes = np.empty((0,9)) 
		for i in range(prev_boxes.shape[0]):
			cur_box = prev_boxes[i]
			if(float(cur_box[DET_SCORE]) > .7 and float(cur_box[POSE_SCORE]) > .7 and cur_box[MATCHING_BOX_ID] == -1):
				cur_box[DET_SCORE] = float(cur_box[DET_SCORE]) - .1
				cur_box[POSE_SCORE] = float(cur_box[POSE_SCORE]) - .1
				cur_box[MATCHING_BOX_ID] = -1
				kept_boxes = np.append(kept_boxes, [cur_box], axis=0)

		#end for i   


		#now add in the boxes from the current frame we want to keep
		for i in xrange(top_conf.shape[0]):
			if(int(float(keep_box[i])) == 1):

				xmin = int(round(top_xmin[i] * image.shape[1]))
				ymin = int(round(top_ymin[i] * image.shape[0]))
				xmax = int(round(top_xmax[i] * image.shape[1]))
				ymax = int(round(top_ymax[i] * image.shape[0]))
				score = top_conf[i]
				label = top_labels[i]
				pose = top_pose[i]
				pose_score = top_poseScore[i]

				box = np.array([[xmin, ymin, xmax, ymax, score, pose, pose_score,label, -1]])
				kept_boxes = np.append(kept_boxes, box, axis=0)
	
		#end for i   

		#save the kept boxes for this frame as the previous boxes for the next frame
		prev_boxes = kept_boxes

		for i in xrange(kept_boxes.shape[0]):
			box = kept_boxes[i]
			name = '%s: %.2f, predicted pose = %d(%.2f)'%(box[LABEL], float(box[DET_SCORE]), int(float(box[POSE])), float(box[POSE_SCORE]))
			coords = (int(float(box[XMIN])), int(float(box[YMIN]))), int(float(box[XMAX])-float(box[XMIN])+1),int(float(box[YMAX])-float(box[YMIN])+1)
			color = colors[i % len(colors)]
			currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
			currentAxis.text(box[XMIN], box[YMIN], name, bbox={'facecolor':'white', 'alpha':0.4})

 
		#END SMOTHING CODE

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

#SMOTHING CODE
def get_bboxes_iou(box1, box2):
	width1 = float(box1[2]) - float(box1[0])	
	height1 = float(box1[3]) - float(box1[1])	
	width2 = float(box2[2]) - float(box2[0])	
	height2 = float(box2[3]) - float(box2[1])	


	int_width = min(float(box1[2]), float(box2[2])) - max(float(box1[0]), float(box2[0]))
	int_height = min(float(box1[3]), float(box2[3])) - max(float(box1[1]), float(box2[1]))


	intersection_area = int_width * int_height
	if(int_width < 0 or int_height < 0):
		intersection_area = 0
	


	union_area = width1*height1 + width2*height2 - intersection_area


	iou = intersection_area / union_area
	return iou

#END SMOTHING CODE


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--vid_seq_id', default=1, type=int, help='int id of vid sequence')
	parser.add_argument('--model', type=str, help='model folder')
	parser.add_argument('--iter', type=int, help='iteration to use ')

	args = parser.parse_args()
	params = vars(args)
	main(params)






