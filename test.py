"""
test.py

Testing the Faster R-CNN model for gene prediction in the genome sequence.

@author: Venkata Pillutla
"""

from keras.utils import generic_utils
from keras.layers import Input
from keras.optimizers import Adam, SGD, RMSprop
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

from utils.preprocessing import one_hot_encoding_sequences
from utils import losses
from models.baseNN import neuralnets as nn
from models.baseNN import Config

from data_generation import data_generators
from data_generation.GenemeDataFetcher import read_genome_sequence_fromFile, writeFeatureToDisk
import data_generation.DataGenerator
from data_generation.DataGenerator import read_genome_fasta
from data_generation import roi_helpers

from numpy import random

import sys
import time

def errorCalc(original, predicted):

	"""
	calculating l1 loss
	"""
	if original.shape[0] > predicted.shape[0]:
		x = original[:predicted.shape[0],:] - predicted
		x = np.sum(x, axis = 0)/predicted.shape[0]

	else:
		x = original - predicted[:original.shape[0],:]
		x = np.sum(x, axis = 0)/original.shape[0]

	
	#print x
	#print np.sum(x)

	return np.sum(x)

	#x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
	#return lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])

def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)


def model_test(test_sequences, allSequences, c):

	Y_test = []

	for seqid in range(10):
		bboxes = allSequences[seqid]['bboxes']

		Y_test.append(bboxes)


	num_features = 512
	class_mapping = c.class_mapping
	bbox_threshold = 0.5

	#ratio = image min side / height
	ratio = c.im_size / 1
	

	print('Testing the model')

	if K.image_dim_ordering() == 'th':
		input_shape_img = (4, None, None)
		input_shape_features = (num_features, None, None)
	else:
		input_shape_img = (None, None, 4)
		input_shape_features = (None, None, num_features)


	img_input = Input(shape=input_shape_img)
	roi_input = Input(shape=(c.num_rois, 4))
	feature_map_input = Input(shape=input_shape_features)

	# define the base network (resnet here, can be VGG, Inception, etc)
	shared_layers = nn.nn_base(img_input, trainable=True)

	# define the RPN, built on the base layers
	num_anchors = len(c.anchor_box_scales) * len(c.anchor_box_ratios)
	rpn_layers = nn.regionProposalNetwork(shared_layers, num_anchors)

	classifier = nn.classifier(feature_map_input, roi_input, c.num_rois, nb_classes=len(class_mapping), trainable=True)

	model_rpn = Model(img_input, rpn_layers)
	model_classifier_only = Model([feature_map_input, roi_input], classifier)

	model_classifier = Model([feature_map_input, roi_input], classifier)

	print('Loading weights from {}'.format(c.model_path))
	model_rpn.load_weights(c.model_path, by_name=True)
	model_classifier.load_weights(c.model_path, by_name=True)

	model_rpn.compile(optimizer='sgd', loss='mse')
	model_classifier.compile(optimizer='sgd', loss='mse')



	original_locations_npy_array = []
	predicted_locations_npy_array = []
	loss = []
	for idx, sequence in enumerate(test_sequences):
		print('sequence id : {}').format(idx)
		st = time.time()
		[Y1, Y2, F] = model_rpn.predict(np.reshape(sequence,(1, 1, 10000, 4)))
		R = roi_helpers.rpn_to_roi(Y1, Y2, c, K.image_dim_ordering(), overlap_thresh=0.1)

		# convert from (x1,y1,x2,y2) to (x,y,w,h)
		R[:, 2] -= R[:, 0]
		R[:, 3] -= R[:, 1]

		#making y2 = 1
		#R[:,3] = 1 

		# apply the spatial pyramid pooling to the proposed regions
		bboxes = {}
		probs = {}
		for jk in range(R.shape[0]//c.num_rois + 1):
			ROIs = np.expand_dims(R[c.num_rois*jk:c.num_rois*(jk+1), :], axis=0)
			if ROIs.shape[1] == 0:
				break

			if jk == R.shape[0]//c.num_rois:
				#pad R
				curr_shape = ROIs.shape
				target_shape = (curr_shape[0],c.num_rois,curr_shape[2])
				ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
				ROIs_padded[:, :curr_shape[1], :] = ROIs
				ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
				ROIs = ROIs_padded

			[P_cls, P_regr] = model_classifier_only.predict([F, ROIs])
			
			#print P_cls.shape

			for ii in range(P_cls.shape[1]):

				#print np.max(P_cls[0, ii, :])
				#print (P_cls.shape[2]-1)
				#if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
				if np.max(P_cls[0, ii, :]) < bbox_threshold:
					continue

				#print P_cls
				class_number = np.argmax(P_cls[0, ii, :])
				#print class_mapping.keys()
				cls_name = class_mapping[class_mapping.keys()[class_number]]
				#cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

				#print cls_name

				if cls_name not in bboxes:
					bboxes[cls_name] = []
					probs[cls_name] = []

				(x, y, w, h) = ROIs[0, ii, :]

				

				cls_num = np.argmax(P_cls[0, ii, :])
				try:
					(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]

					
					
					tx /= c.classifier_regr_std[0]
					ty /= c.classifier_regr_std[1]
					tw /= c.classifier_regr_std[2]
					th /= c.classifier_regr_std[3]
					
					
					x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
					



					#print('Bounding box {}').format((x,y,w,h))
				except:
					pass
				
				bboxes[cls_name].append([c.rpn_stride*x, c.rpn_stride*y, c.rpn_stride*(x+w), c.rpn_stride*(y+h)])
				
				#using x1, y1, x2, y2 instead of x1, y1, w, h	
				#bboxes[cls_name].append([c.rpn_stride*x, c.rpn_stride*y, c.rpn_stride*(x+w), c.rpn_stride*(y+h)])

				probs[cls_name].append(np.max(P_cls[0, ii, :]))
		all_dets = []



		gene_locations = []
		for key in bboxes:
			bbox = np.array(bboxes[key])
			new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.1)

			for jk in range(new_boxes.shape[0]):
				(x1, y1, x2, y2) = new_boxes[jk,:]

				(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
				
				if int(key) == 1:
					gene_locations.append([real_x1, real_y1, real_x2, real_y2])
				#cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

				textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
				print('gene location: {}').format((real_x1, real_y1, real_x2, real_y2))
				print textLabel
				all_dets.append((key,100*new_probs[jk]))

				#(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
				#textOrg = (real_x1, real_y1-0)

		
		original_gene_locations = []
		for dic in Y_test[idx]:
			if dic.get('class')!='bg':
				original_gene_locations.append([dic.get('x1'), dic.get('y1'), dic.get('x2'), dic.get('y2')])

		gene_locations = np.array(gene_locations)
		original_gene_locations = np.array(original_gene_locations)

		
		gene_locations = np.sort(gene_locations.view('i8,i8,i8,i8'), order=['f1'], axis=0).view(np.int)
		original_gene_locations = np.sort(original_gene_locations.view('i8,i8,i8,i8'), order=['f1'], axis=0).view(np.int)

		print('predicted gene_locations, {0}').format(gene_locations)
		print('original sequence, {0}').format(original_gene_locations)
		
		original_locations_npy_array.append(original_gene_locations)
		predicted_locations_npy_array.append(gene_locations)

		#np.save('original_gene_locations.npy',original_gene_locations)
		#np.save('predicted_gene_locations.npy',gene_locations)

		l1_loss = errorCalc(original_gene_locations, gene_locations)
		loss.append(loss)
		print('Testing L1 loss: {}').format(l1_loss)
		print('Elapsed time = {}'.format(time.time() - st))
		#print(all_dets)


	np.save('original_gene_locations.npy',np.array(original_locations_npy_array))
	np.save('predicted_gene_locations.npy',np.array(predicted_locations_npy_array))
	#print('total test loss').format(np.sum(np.array(loss))//len(test_sequences))

