"""
train.py

Faster R-CNN for gene prediction in the genome sequence.

@author: Venkata Pillutla
"""

from keras.utils import generic_utils
from keras.layers import Input
from keras.optimizers import Adam, SGD, RMSprop
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences


from utils import losses
from utils.preprocessing import one_hot_encoding_sequences
from models.baseNN import neuralnets
from models.baseNN import Config
from data_generation import data_generators
from data_generation.GenemeDataFetcher import read_genome_sequence_fromFile, writeFeatureToDisk
import data_generation.DataGenerator
from data_generation.DataGenerator import read_genome_fasta, getData
from data_generation import roi_helpers

from sklearn.model_selection import train_test_split

from numpy import random

import sys
import time

from test import model_test


#sys.stdout = open('outputaugten.txt', 'w')

np.random.seed(9)
sys.setrecursionlimit(40000)

c = Config.Config()

#number of epochs
num_epochs = c.num_of_epochs

counter = 1
gene_location_dict = {}
genome_ids = ['511145.12', '100226.15', '107806.10', '1028307.3']
#whole sequence splitted into multiple smaller sequences.
sequence = []

class_count = {}
class_mapping = {}

allSequences = {}

num_of_bg_instances = 0

for genome_id in genome_ids[0:3]:
	#genome_id = genome_ids[0]


	wholeSequence = read_genome_fasta(str(genome_id))
	dataFrame = getData(str(genome_id))


	for index, row in dataFrame.iterrows():
		#gene_start <=10000 and gene_end <=10000
		if int(row['start']) <= (counter * c.seqlen) and int(row['end'])<= (counter * c.seqlen):
			if counter-1 not in gene_location_dict.keys():
				gene_location_dict[counter-1] = []
			gene_location_dict[counter-1].append((row['start'],row['end']))

		#gene starts before 10000 but gene ends after 10000 characters
		elif int(row['start']) <= (counter * c.seqlen) and (int(row['end']) > (counter * c.seqlen)):
			if counter-1 not in gene_location_dict.keys():
				gene_location_dict[counter-1] = []
			


			gene_location_dict[counter-1].append((row['start'],counter*c.seqlen))
			#placing the ending portion of the gene in the next sequence
			gene_location_dict[counter] = []
			gene_location_dict[counter].append(((counter)*c.seqlen, row['end']))
			counter = counter + 1
		elif int(row['start']) > (counter *c.seqlen) and (int(row['start'] < (2 * counter * c.seqlen))):
			counter = counter +1
			if (int(row['end']) > (counter * c.seqlen)) and (int(row['end'] < (2 * counter * c.seqlen))):
				if counter-1 not in gene_location_dict.keys():
					gene_location_dict[counter-1] = []
				gene_location_dict[counter-1].append((row['start'], row['end']))


	#dividing the complete genome sequence into strings of length = c.seqlen
	for i in range(0, len(wholeSequence), c.seqlen):
		sequence.append(wholeSequence[i: i+c.seqlen])

	try:
		#last element null - issue should be fixed instead of hard coding
		for seqid, seq in enumerate(sequence[0:-1]):

			#print 'currently processing '+str(seqid)+' of '+str(len(sequence)-1)+' sequences.'
			if seqid not in allSequences.keys():



				
				a = []
				a.append(seq)
				a = one_hot_encoding_sequences(a, c.seqlen)
				a = np.array(a)

				
				try:
					a = a.reshape((1, c.seqlen, 4))
				except:
					#check
					continue
				allSequences[seqid] = {}

				allSequences[seqid]['seqid'] = seqid
				
				allSequences[seqid]['sequence'] = a
				allSequences[seqid]['width'] = c.seqlen
				allSequences[seqid]['height'] = 1
				allSequences[seqid]['bboxes'] = []
				if np.random.randint(0,6) > 0:
					allSequences[seqid]['seqset'] = 'trainval'
				else:
					allSequences[seqid]['seqset'] = 'test'

				#adding ground truth bounding boxes
				if seqid in gene_location_dict.keys():
					
					#no gene sequence in this portion of the genome
					if len(gene_location_dict[seqid]) == 0:
						print('INFO: sequenceid {0} doesnt contain a gene.').format(seqid)
						class_name = 'bg'
						if class_name not in class_count:
							class_count[class_name] = 1
						else:
							class_count[class_name] += 1

						if class_name not in class_mapping:
							class_mapping[class_name] = len(class_mapping)	

					#gene exists in this portion of the genome					
					else:
						previous_x1 = 0
						previous_x2 = 0
						for bbox in gene_location_dict[seqid]:
							class_name = 1
							
							if seqid > 0:
								#class = 1
								current_x1 = int(bbox[0])-(seqid * c.seqlen)
								current_x2 = int(bbox[1])-(seqid * c.seqlen)
								if current_x1 <0 or current_x1 >c.seqlen or current_x2 <0 or current_x2 >c.seqlen:
									#print seqid
									#print('case 1 : {0}, {1}').format(current_x1, current_x2)
									continue
								allSequences[seqid]['bboxes'].append({'class': int(1), 'x1': current_x1 , 'x2': current_x2, 'y1': int(0),'y2': int(1)})
								

							else:
								current_x1 = int(bbox[0])
								current_x2 = int(bbox[1])
								if current_x1 <0 or current_x1 >c.seqlen or current_x2 <0 or current_x2 >c.seqlen:
									#print seqid
									#print('case 2 : {0}, {1}').format(current_x1, current_x2)
									continue
								allSequences[seqid]['bboxes'].append({'class': int(1), 'x1': current_x1 , 'x2': current_x2, 'y1': int(0),'y2': int(1)})
							
							
							#creating background bboxes
							if previous_x2 > 0 and current_x1 > previous_x2:
								num_of_bg_instances+=1
								allSequences[seqid]['bboxes'].append({'class': 'bg', 'x1': previous_x2 , 'x2': current_x1, 'y1': int(0),'y2': int(1)})
								if 'bg' not in class_count:
									class_count['bg'] = 1
								else:
									class_count['bg'] += 1

							previous_x2 = current_x2
							previous_x1 = current_x1

							if class_name not in class_count:
								class_count[class_name] = 1
							else:
								class_count[class_name] += 1

							if class_name not in class_mapping:
								class_mapping[class_name] = len(class_mapping)

				#allSequences[seqid]['bboxes'].append({'class': int(class_name), 'x1': int(x1) , 'x2': int(x2), 'y1': int(y1),'y2': int(y2)})


	except Exception as e:
		print(e)


class_mapping['bg'] = len(class_mapping)


print('class_mapping {}').format(class_mapping)
##each data point is a dictionary with seqid starting from 0, sequence, height = 1, width = c.seqlen, bboxes = list of single item dict


print('total num of instances = {}').format(len(allSequences))
print('num of background instances = {}').format(num_of_bg_instances)



all_data = []
for key in allSequences:
	all_data.append(allSequences[key])

random.shuffle(all_data)
num_sequences = len(all_data)

train_seqs = [s for s in all_data if s['seqset'] == 'trainval']
val_seqs = [s for s in all_data if s['seqset'] == 'test']

print('Num train samples {}'.format(len(train_seqs)))
print('Num val samples {}'.format(len(val_seqs)))


#all_data_npy_format = np.array(all_data)
#print('INFO: Storing data on to disk')
#np.save('sample_data/data_set.npy', all_data_npy_format)
#all_data = np.load('sample_data/data_set.npy')


#getting anchor boxes with gt labels
#data_gen_train = data_generators.get_anchor_gt(all_data, class_count, c, neuralnets.get_img_output_length, K.image_dim_ordering(), mode='train')
data_gen_train = data_generators.get_anchor_gt(train_seqs, class_count, c, neuralnets.get_img_output_length, K.image_dim_ordering(), mode='train')
data_gen_val = data_generators.get_anchor_gt(val_seqs, class_count, c, neuralnets.get_img_output_length,K.image_dim_ordering(), mode='val')

sequenceWidth = len(sequence[0])

X = np.array(one_hot_encoding_sequences(sequence, c.seqlen))

X = X.reshape(X.shape[0], 1, sequenceWidth, 4)

np.random.shuffle(X)

X_test = X[0:10]
X = X[10:]
c.class_mapping = class_mapping

#print allSequences[0]['bboxes']
#model_test(X_test, allSequences, c)
#exit(0)

genome_shape = (1, sequenceWidth, 4)

# genome sequence input has 4 channels since the images are one hot encoded with 'atgc'
genome_sequence_input = Input(shape = genome_shape)

roi_input = Input(shape = (None, 4))

number_of_anchors = len(c.anchor_box_scales) * len(c.anchor_box_ratios)

#define the neural network here
shared_layers = neuralnets.nn_base(genome_sequence_input, trainable = True)


#define the Region proposal network here - TODO
rpn = neuralnets.regionProposalNetwork(shared_layers, number_of_anchors)

#classifier = neuralnets.classifier(shared_layers, roi_input, c.num_rois, nb_classes=len(class_mapping.keys(), trainable=True)
#check if bg should be counted or not
classifier = neuralnets.classifier(shared_layers, roi_input, c.num_rois, nb_classes=len(class_count), trainable=True)

#print('rpn output {0} , {1}').format(rpn[0].shape, rpn[1].shape)


model_rpn = Model(genome_sequence_input, rpn[:2])
#model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([genome_sequence_input, roi_input], classifier)


# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([genome_sequence_input, roi_input], rpn[:2] + classifier)


optimizer = Adam(lr=1e-4)
optimizer_classifier = Adam(lr=1e-4)

model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(number_of_anchors), losses.rpn_loss_regr(number_of_anchors)])


#model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(class_mapping.keys())-1)], metrics={'dense_class_{}'.format(len(class_count)): 'accuracy'})
model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(class_count)-1)], metrics={'dense_class_{}'.format(len(class_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')

#c.epoch_length = c.c.epoch_length
iter_num = 0

losses = np.zeros((c.epoch_length, 5))
losses_val = np.zeros((c.epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []

rpn_accuracy_rpn_monitor_val = []
rpn_accuracy_for_epoch_val = []
start_time = time.time()
best_loss = np.Inf
best_loss_val = np.Inf

print('INFO: Model RPN summary')
print(model_rpn.summary())
print('INFO: Model classifier summary')
print(model_classifier.summary())

class_mapping_inv = {v: k for k, v in class_mapping.items()}

rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []


print("INFO: -----Training Start-----")
for epoch_num in range(num_epochs):

	progbar = generic_utils.Progbar(c.epoch_length, width = 20)
	print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))
	
	while True:
		if len(rpn_accuracy_rpn_monitor) == c.epoch_length and c.verbose:
			mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
			rpn_accuracy_rpn_monitor = []
			print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, c.epoch_length))
			if mean_overlapping_bboxes == 0:
				print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

		X, Y, img_data = next(data_gen_train)

		X_val, Y_val, img_data_val = next(data_gen_val)
		

		X = X.reshape(1, 1, sequenceWidth, 4)
		

		Y[1] = np.transpose(Y[1], (0, 2, 3, 1))
		Y[0] = np.transpose(Y[0], (0, 2, 3, 1))

		X_val = X.reshape(1, 1, sequenceWidth, 4)
		Y_val[1] = np.transpose(Y_val[1], (0, 2, 3, 1))
		Y_val[0] = np.transpose(Y_val[0], (0, 2, 3, 1))


		
		try:
			loss_rpn = model_rpn.train_on_batch(X, [Y[0], Y[1]])
			P_rpn = model_rpn.predict_on_batch(X)

			loss_rpn_val = model_rpn.evaluate(X_val, [Y_val[0], Y_val[1]])
			P_rpn_val = model_rpn.predict_on_batch(X_val)
			
			#R contains boxes and probabilties.
			R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], c, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
			R_val = roi_helpers.rpn_to_roi(P_rpn_val[0], P_rpn_val[1], c, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)

			# note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
			#creates a training set to train the classifier and regressor
			#x2 is the roi, y1 is class label, y2 is the bounding box coordinates
			X2, Y1, Y2, Ious = roi_helpers.calc_iou(R, img_data, c, class_mapping)
			X2_val, Y1_val, Y2_val, Ious_val = roi_helpers.calc_iou(R_val, img_data_val, c, class_mapping)

			if X2 is None:
				rpn_accuracy_rpn_monitor.append(0)
				rpn_accuracy_for_epoch.append(0)
				continue

			neg_samples = np.where(Y1[0, :, -1] == 1)
			pos_samples = np.where(Y1[0, :, -1] == 0)


			if len(neg_samples) > 0:
				neg_samples = neg_samples[0]
			else:
				neg_samples = []

			if len(pos_samples) > 0:
				pos_samples = pos_samples[0]
			else:
				pos_samples = []

			rpn_accuracy_rpn_monitor.append(len(pos_samples))
			rpn_accuracy_for_epoch.append((len(pos_samples)))

			if c.num_rois > 1:
				if len(pos_samples) < c.num_rois//2:
					selected_pos_samples = pos_samples.tolist()
				else:
					selected_pos_samples = np.random.choice(pos_samples, c.num_rois//2, replace=False).tolist()
				try:
					selected_neg_samples = np.random.choice(neg_samples, c.num_rois - len(selected_pos_samples), replace=False).tolist()
				except:
					selected_neg_samples = np.random.choice(neg_samples, c.num_rois - len(selected_pos_samples), replace=True).tolist()

				sel_samples = selected_pos_samples + selected_neg_samples
			else:
				# in the extreme case where num_rois = 1, we pick a random pos or neg sample
				selected_pos_samples = pos_samples.tolist()
				selected_neg_samples = neg_samples.tolist()
				if np.random.randint(0, 2):
					sel_samples = random.choice(neg_samples)
				else:
					sel_samples = random.choice(pos_samples)


			#redundant code
			neg_samples_val = np.where(Y1_val[0, :, -1] == 1)
			pos_samples_val = np.where(Y1_val[0, :, -1] == 0)


			if len(neg_samples_val) > 0:
				neg_samples_val = neg_samples_val[0]
			else:
				neg_samples_val = []

			if len(pos_samples_val) > 0:
				pos_samples_val = pos_samples_val[0]
			else:
				pos_samples_val = []

			rpn_accuracy_rpn_monitor_val.append(len(pos_samples_val))
			rpn_accuracy_for_epoch_val.append((len(pos_samples_val)))

			if c.num_rois > 1:
				if len(pos_samples_val) < c.num_rois//2:
					selected_pos_samples_val = pos_samples_val.tolist()
				else:
					selected_pos_samples_val = np.random.choice(pos_samples_val, c.num_rois//2, replace=False).tolist()
				try:
					selected_neg_samples_val = np.random.choice(neg_samples_val, c.num_rois - len(selected_pos_samples_val), replace=False).tolist()
				except:
					selected_neg_samples_val = np.random.choice(neg_samples_val, c.num_rois - len(selected_pos_samples_val), replace=True).tolist()

				sel_samples_val = selected_pos_samples_val + selected_neg_samples_val
			else:
				# in the extreme case where num_rois = 1, we pick a random pos or neg sample
				selected_pos_samples_val = pos_samples_val.tolist()
				selected_neg_samples_val = neg_samples_val.tolist()
				if np.random.randint(0, 2):
					sel_samples_val = random.choice(neg_samples_val)
				else:
					sel_samples_val = random.choice(pos_samples_val)



			loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

			
			

			loss_class_val = model_classifier.evaluate([X_val, X2_val[:,sel_samples_val,:]], [Y1_val[:,sel_samples_val,:], Y2_val[:,sel_samples_val,:]])


			losses[iter_num, 0] = loss_rpn[1]
			losses[iter_num, 1] = loss_rpn[2]
			losses[iter_num, 2] = loss_class[0]
			losses[iter_num, 3] = loss_class[1]
			losses[iter_num, 4] = loss_class[2]

			losses_val[iter_num, 0] = loss_rpn_val[1]
			losses_val[iter_num, 1] = loss_rpn_val[2]
			losses_val[iter_num, 2] = loss_class_val[0]
			losses_val[iter_num, 3] = loss_class_val[1]
			losses_val[iter_num, 4] = loss_class_val[2]


			iter_num += 1

			#progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
			#						  ('detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 3]))])

			progbar.update(iter_num, [('\n rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
									  ('\n detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 3])), 
									  ('\n val_rpn_cls', np.mean(losses_val[:iter_num, 0])), ('val_rpn_regr', np.mean(losses_val[:iter_num, 1])),
									  ('\n val_detector_cls', np.mean(losses_val[:iter_num, 2])), ('val_detector_regr', np.mean(losses_val[:iter_num, 3]))])


			if iter_num == c.epoch_length:

				loss_rpn_cls = np.mean(losses[:, 0])
				loss_rpn_regr = np.mean(losses[:, 1])
				loss_class_cls = np.mean(losses[:, 2])
				loss_class_regr = np.mean(losses[:, 3])
				class_acc = np.mean(losses[:, 4])

				loss_rpn_cls_val = np.mean(losses_val[:, 0])
				loss_rpn_regr_val = np.mean(losses_val[:, 1])
				loss_class_cls_val = np.mean(losses_val[:, 2])
				loss_class_regr_val = np.mean(losses_val[:, 3])
				class_acc_val = np.mean(losses_val[:, 4])

				mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
				rpn_accuracy_for_epoch = []


				mean_overlapping_bboxes_val = float(sum(rpn_accuracy_for_epoch_val)) / len(rpn_accuracy_for_epoch_val)
				rpn_accuracy_for_epoch_val = []

				if c.verbose:
					#training results
					print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
					print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
					print('Loss RPN classifier: {}'.format(loss_rpn_cls))
					print('Loss RPN regression: {}'.format(loss_rpn_regr))
					print('Loss Detector classifier: {}'.format(loss_class_cls))
					print('Loss Detector regression: {}'.format(loss_class_regr))

					#validation results
					print('Mean number of bounding boxes from RPN overlapping ground truth boxes for validation: {}'.format(mean_overlapping_bboxes_val))
					print('Classifier accuracy for bounding boxes from RPN for validation: {}'.format(class_acc_val))
					print('Loss RPN classifier for validation: {}'.format(loss_rpn_cls_val))
					print('Loss RPN regression for validation: {}'.format(loss_rpn_regr_val))
					print('Loss Detector classifier for validation: {}'.format(loss_class_cls_val))
					print('Loss Detector regression for validation: {}'.format(loss_class_regr_val))

					time_taken = time.time() - start_time
					print('Elapsed time: {}').format(time_taken)
					fil = open('without_ran_time.csv','a')
					st = str(epoch_num+1)+','+str(time_taken)+'\n'
					fil.write(st)
					fil.close()

				curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
				curr_loss_val = loss_rpn_cls_val + loss_rpn_regr_val + loss_class_cls_val + loss_class_regr_val
				iter_num = 0
				start_time = time.time()

				if curr_loss_val < best_loss_val:
					if c.verbose:
						print('Total validation loss decreased from {} to {}, saving weights'.format(best_loss_val,curr_loss_val))
					best_loss_val = curr_loss_val
					#model_all.save_weights(c.model_path)


				if curr_loss < best_loss:
					if c.verbose:
						print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
					best_loss = curr_loss
					model_all.save_weights(c.model_path)

				break
		except Exception as e:
			#print('Error in training : ')
			#print(e)
			continue
			#break
	#model_test(X_test, allSequences, c)

print('Training Complete, exiting.')



