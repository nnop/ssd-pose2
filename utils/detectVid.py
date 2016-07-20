import matplotlib
#matplotlib.use('Agg')
#import cv2
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

	mod_base = '/net/bvisionserver1/playpen2/poirson/ssd-pose/models/VGGNet/philData/'
	
	mod_idx = args['model']
	iterx = args['iter']

	model_def = osp.join(mod_base, mod_idx, 'deploy.prototxt')
	model_weights = 'VGG_Rohit_%s_iter_%d.caffemodel' % (mod_idx, iterx)
	#model_weights = 'VGG_Pascal3D_%s_iter_%d.caffemodel' % (mod_idx, iterx)

	mod_file = osp.join(mod_base, mod_idx, model_weights)

	net = caffe.Net(model_def, mod_file, caffe.TEST)

	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2, 0, 1))
	transformer.set_mean('data', np.array([104,117,123])) # mean pixel
	transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
	transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

	scene_out = 'video_sequence_%d_smooth=%r_%s' % (args['vid_seq_id'], args['smooth'], mod_idx)
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

	
	#prev_boxes = np.empty((0,9)) 
	prev_boxes = []
	#END SMOTHING CODE

	for idx, fname in enumerate(seq):
		if idx % 20 == 0:
			print '%d/%d processed' % (idx, len(seq))
		#plt.figure(figsize=(20,15))

		imfile = osp.join(basePath, scene, seq[idx])
		image_resize = 300
		net.blobs['data'].reshape(1,3,image_resize,image_resize)

		image = caffe.io.load_image(imfile)
		#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		#im_copy = image.copy()
		#plt.imshow(image)

		transformed_image = transformer.preprocess('data', image)
		net.blobs['data'].data[...] = transformed_image


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


		colors = ['b', 'g', 'r', 'c', 'm', 'y']
		#colors = ['b', 'g', 'c', 'm', 'y']
		clr2rgb = {'b': (0,0,255), 'g':(0,255,0), 'r':(255,0,0), 'c':(0,255,255), 'm':(255,0,255), 'y':(255,255,0)}

		plt.imshow(image)
		currentAxis = plt.gca()

		if args['smooth']:

			#keep_box = [0]*len(xrange(top_conf.shape[0]))
			keep_box = np.zeros_like(top_conf)



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
				# GET the previous box which overlaps the most with current box
				for jl in range(len(prev_boxes)):
				#for jl in range(prev_boxes.shape[0]):
					pos_box = prev_boxes[jl]
					#print pos_box.shape
					if(pos_box[LABEL] != label):
						continue
					cur_iou = get_bboxes_iou( [xmin, ymin, xmax, ymax], pos_box[:4])

					if(cur_iou > max_iou):
						max_iou = cur_iou
						prev_box = pos_box
						prev_box_ind = jl

				
				#if this box has a previous matching box 
				if(max_iou > .4):
					#if this is the first box in this frame to match with this box	
					if(prev_box[MATCHING_BOX_ID] == -1):
						prev_box[MATCHING_BOX_ID] = i
						prev_boxes[prev_box_ind] = prev_box
						keep_box[i] = 1
					else:
						omi = prev_box[MATCHING_BOX_ID]

						if(top_conf[omi] < score):
							prev_box[MATCHING_BOX_ID] = i
							prev_boxes[prev_box_ind] = prev_box
							keep_box[i] = 1
							keep_box[omi] = 0

					if(prev_box[MATCHING_BOX_ID] == i):
						if(abs(pose - float(prev_box[POSE])) > 1):
							top_pose[i] = float(prev_box[POSE])
				else:
					prev_box = []
					if(score > .7):
						keep_box[i] = 1 




			#now see if there are any prev boxes that should be kept
			kept_boxes = []
			for i in range(len(prev_boxes)):
			#for i in range(prev_boxes.shape[0]):
				cur_box = prev_boxes[i]
				if(float(cur_box[DET_SCORE]) > .7 and float(cur_box[POSE_SCORE]) > .7 and cur_box[MATCHING_BOX_ID] == -1):
					cur_box[DET_SCORE] = float(cur_box[DET_SCORE]) - .1
					cur_box[POSE_SCORE] = float(cur_box[POSE_SCORE]) - .1
					cur_box[MATCHING_BOX_ID] = -1
					kept_boxes.append( cur_box)

			#end for i   


			#now add in the boxes from the current frame we want to keep
			for i in range(top_conf.shape[0]):
				if(keep_box[i] == 1):

					xmin = int(round(top_xmin[i] * image.shape[1]))
					ymin = int(round(top_ymin[i] * image.shape[0]))
					xmax = int(round(top_xmax[i] * image.shape[1]))
					ymax = int(round(top_ymax[i] * image.shape[0]))
					score = top_conf[i]
					label = top_labels[i]
					pose = top_pose[i]
					pose_score = top_poseScore[i]

					box = [xmin, ymin, xmax, ymax, score, pose, pose_score,label, -1]
					kept_boxes.append(box)
		
			#end for i   

			#save the kept boxes for this frame as the previous boxes for the next frame
			prev_boxes = kept_boxes

			for i in range(len(kept_boxes)):
				box = kept_boxes[i]
				name = '%s: %.2f, pose = %d (%.2f)'%(box[LABEL], float(box[DET_SCORE]), int(float(box[POSE])), float(box[POSE_SCORE]))
				coords = (box[XMIN], box[YMIN]), box[XMAX]-box[XMIN]+1, box[YMAX]-box[YMIN]+1
				color = colors[i % len(colors)]
				currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
				if box[LABEL] != 'diningtable':
					currentAxis.text(box[XMIN], box[YMIN]-3, name, bbox={'facecolor':'white', 'alpha':.3})
				else:
					name = '%s: %.2f, pose = %d (%.2f)'%('table', float(box[DET_SCORE]), int(float(box[POSE])), float(box[POSE_SCORE]))
					currentAxis.text(box[XMIN], box[YMIN]-15, name, fontsize=15, bbox={'facecolor':'white', 'alpha':1.0})

	 
			#END SMOTHING CODE
		else:
			for i in xrange(top_conf.shape[0]):
				xmin = int(round(top_xmin[i] * image.shape[1]))
				ymin = int(round(top_ymin[i] * image.shape[0]))
				xmax = int(round(top_xmax[i] * image.shape[1]))
				ymax = int(round(top_ymax[i] * image.shape[0]))
				score = top_conf[i]
				label = top_labels[i]
				label_ind = top_label_indices[i]
				pose = top_pose[i]
				pose_score = top_poseScore[i]
				name = '%s: %.2f, pose = %d (%.2f)'%(label, score, int(pose), pose_score)
				coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
				if label != 'diningtable':
					currentAxis.text(xmin, ymin-3, name, bbox={'facecolor':'white', 'alpha':.3})
				else:
					name = '%s: %.2f, pose = %d (%.2f)'%('table', score, int(pose), pose_score)
					currentAxis.text(xmin, ymin-15, name, fontsize=15, bbox={'facecolor':'white', 'alpha':1.0})
				#name = '%d, %.2f, %d, %.2f' % (label_ind, score, pose, pose_score)
				
				color = colors[i % len(colors)]
				currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
				#clr = clr2rgb[color]
				#cv2.rectangle(image, (xmin, ymin), (xmax, ymax), clr, 2)
				#cv2.putText(image, name, (xmin-5, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255))
				

		#alpha = 1
		#cv2.addWeighted(image, alpha, im_copy, (1-alpha), 0, im_copy)
		#cv2.imshow("show", im_copy)
		#cv2.waitKey()
		#cv2.destroyAllWindows()
		plt.axis('off')
		plt.tight_layout()
		plt.savefig(osp.join(outDir, fname), bbox_inches='tight', pad_inches = 0)
		plt.close()
		#plt.show()
		#raw_input("Press Enter to continue...")
		#if idx == 5:
		#	break

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


def get_bboxes_iou(box1, box2):
	box1 = [float(val) for val in box1]
	box2 = [float(val) for val in box2]

	width1 = box1[2] - box1[0]
	height1 = box1[3] - box1[1]
	width2 = box2[2] - box2[0]
	height2 = box2[3] - box2[1]	

	int_width = min(box1[2], box2[2]) - max(box1[0], box2[0])
	int_height = min(box1[3], box2[3]) - max(box1[1], box2[1])


	intersection_area = int_width * int_height
	if(int_width < 0 or int_height < 0):
		intersection_area = 0

	union_area = width1*height1 + width2*height2 - intersection_area

	iou = intersection_area / union_area
	return iou



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--vid_seq_id', default=1, type=int, help='int id of vid sequence')
	parser.add_argument('--model', type=str, help='model folder')
	parser.add_argument('--iter', type=int, help='iteration to use ')
	parser.add_argument('--smooth', action='store_true', help='smooth outputs')

	args = parser.parse_args()
	params = vars(args)
	main(params)






