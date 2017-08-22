#from __future__ import absolute_import

from keras import backend as keras
from keras.layers import Activation, Add, Dense, Flatten, AveragePooling2D, ZeroPadding2D, Conv2D, Input, Embedding,TimeDistributed, Convolution2D
from keras.models import Sequential
from keras.layers.pooling import MaxPooling2D
import keras.backend as K

from RoiPoolingConv import RoiPoolingConv
from FixedBatchNormalization import FixedBatchNormalization
#Base convolution layer for generating feature map from genome sequence.

def get_img_output_length(width, height):
	def get_output_length(input_length):
		# zero_pad
		input_length += 6
		# apply 4 strided convolutions
		#filter_sizes = [7, 3, 1, 1]
		filter_sizes = [6]
		stride = 2
		for filter_size in filter_sizes:
			input_length = (input_length - filter_size + stride) // stride
		return input_length

	#return get_output_length(width), 1
	#return 4836, 1
	return 9774, 1

def regionProposalNetwork(base_layers, noOfAnchors):
	"""
	Region Proposal Network
	"""
	x = Conv2D(512, (1, 300), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)
	print 'INFO: rpn_conv1: ',x



	#x = Conv2D(512, (1, 302), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv2')(base_layers)
	#x = MaxPooling2D((1,2), strides = (1,2))(x)

	x_class = Conv2D(noOfAnchors, (1, 103), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
	print 'INFO: rpn_out_class: ',x_class
	x_regr = Conv2D(noOfAnchors * 4, (1, 103), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)
	print 'INFO: rpn_out_regress: ',x_regr
	return [x_class, x_regr, base_layers]


def nn_base(input_tensor = None, trainable = True):
	input_shape = (None, None, 4)

	if input_tensor is None:
		print 'DEBUG: input tensor is none.'
		img_input = Input(shape=input_shape)
	else:
		if not K.is_keras_tensor(input_tensor):
			print "DEBUG: Input not a keras tensor. Converting it to Keras Tensor"
			img_input = Input(tensor = input_tensor, shape=input_shape)
		else:
			img_input = input_tensor


	#sequence_input = Input(tensor = img_input, shape = input_shape, name = 'main_input')
	#32 2x1 filters
	sequence_input = input_tensor
	x = Conv2D(3, (1, 125), activation = 'relu', strides = 1)(sequence_input)
	print 'INFO: Base Layer conv1 ', x
	x = FixedBatchNormalization(axis = 3)(x)
	print 'INFO: Base Layer Fixed Batch Normalization: ', x
	x = Activation('relu')(x)
	#x = MaxPooling2D((1,2), strides = (1,2))(x)
	"""
	x = Conv2D(24, (1, 125), activation = 'relu', strides = 1)(x)
	x = Activation('relu')(x)
	x = MaxPooling2D((1,2), strides = (1,2))(x)

	x = Conv2D(24, (1, 125), activation = 'relu', strides = 1)(x)

	"""
	
	x = conv_block(x, 3, [128,128,256], stage = 2, block = 'a', strides = (1,1), trainable = True)
	x = identity_block(x, 3, [128, 128, 256], stage=2, block='b', trainable = trainable)
	x = identity_block(x, 3, [128, 128, 256], stage=2, block='c', trainable = trainable)
	
	#x = MaxPooling2D((1,2), strides = (1,2))(x)


	"""
	x = Conv2D(3, (1, 470), activation = 'relu', strides = 1)(x)
	print 'INFO: Base Layer conv1 ', x
	x = FixedBatchNormalization(axis = 3)(x)
	print 'INFO: Base Layer Fixed Batch Normalization: ', x
	x = Activation('relu')(x)
	

	x = Conv2D(3, (1, 649), activation = 'relu', strides = 1)(x)
	print 'INFO: Base Layer conv1 ', x
	x = FixedBatchNormalization(axis = 3)(x)
	print 'INFO: Base Layer Fixed Batch Normalization: ', x
	x = Activation('relu')(x)
	"""
	

	x = conv_block(x, 3, [256,256,512], stage = 3, block = 'a', strides = (1,1), trainable = True)
	x = identity_block(x, 3, [256, 256, 512], stage=3, block='b', trainable = trainable)
	x = identity_block(x, 3, [256, 256, 512], stage=3, block='c', trainable = trainable)
	
	"""
	x = conv_block(x, 3, [128, 128, 512], stage=4, block='a', trainable = trainable)
	x = identity_block(x, 3, [128, 128, 512], stage=4, block='b', trainable = trainable)
	x = identity_block(x, 3, [128, 128, 512], stage=4, block='c', trainable = trainable)
	#x = identity_block(x, 3, [128, 128, 512], stage=4, block='d', trainable = trainable)

	
	#x = MaxPooling2D((1,2), strides = (1,2))(x)

	x = Conv2D(3, (1, 128), activation = 'relu', strides = 1)(x)
	print 'INFO: Base Layer conv1 ', x
	x = FixedBatchNormalization(axis = 3)(x)
	print 'INFO: Base Layer Fixed Batch Normalization: ', x
	x = Activation('relu')(x)

	x = Conv2D(3, (1, 128), activation = 'relu', strides = 1)(x)
	print 'INFO: Base Layer conv1 ', x
	x = FixedBatchNormalization(axis = 3)(x)
	print 'INFO: Base Layer Fixed Batch Normalization: ', x
	x = Activation('relu')(x)
	

	x = Conv2D(3, (1, 128), activation = 'relu', strides = 1)(x)
	print 'INFO: Base Layer conv1 ', x
	x = FixedBatchNormalization(axis = 3)(x)
	print 'INFO: Base Layer Fixed Batch Normalization: ', x
	x = Activation('relu')(x)

	#x = identity_block(x, 3, [128, 256, 256], stage=4, block='b', trainable = trainable)

	x = Conv2D(3, (1, 128), activation = 'relu', strides = 1)(x)
	print 'INFO: Base Layer conv1 ', x
	x = FixedBatchNormalization(axis = 3)(x)
	print 'INFO: Base Layer Fixed Batch Normalization: ', x
	x = Activation('relu')(x)

	"""
	print 'INFO: Base Layer output ', x




	return x


def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True):

	nb_filter1, nb_filter2, nb_filter3 = filters
	
	if K.image_dim_ordering() == 'tf':
		bn_axis = 3
	else:
		bn_axis = 1

	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Convolution2D(nb_filter1, (1, 1), name=conv_name_base + '2a', trainable=trainable)(input_tensor)
	x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Convolution2D(nb_filter2, (1, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable)(x)
	x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
	x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	x = Add()([x, input_tensor])
	x = Activation('relu')(x)
	return x


def identity_block_td(input_tensor, kernel_size, filters, stage, block, trainable=True):

	# identity block time distributed
	nb_filter1, nb_filter2, nb_filter3 = filters
	if K.image_dim_ordering() == 'tf':
		bn_axis = 3
	else:
		bn_axis = 1

	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = TimeDistributed(Convolution2D(nb_filter1, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2a')(input_tensor)
	x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = TimeDistributed(Convolution2D(nb_filter2, (1, kernel_size), trainable=trainable, kernel_initializer='normal',padding='same'), name=conv_name_base + '2b')(x)
	x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2c')(x)
	x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

	x = Add()([x, input_tensor])
	x = Activation('relu')(x)

	return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(1, 2), trainable=True):

	nb_filter1, nb_filter2, nb_filter3 = filters
	if K.image_dim_ordering() == 'tf':
		bn_axis = 3
	else:
		bn_axis = 1

	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Convolution2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', trainable=trainable)(input_tensor)
	x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Convolution2D(nb_filter2, (1, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable)(x)
	x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
	x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	shortcut = Convolution2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', trainable=trainable)(input_tensor)
	shortcut = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

	x = Add()([x, shortcut])
	x = Activation('relu')(x)
	return x


def conv_block_td(input_tensor, kernel_size, filters, stage, block, input_shape, strides=(2, 2), trainable=True):

	# conv block time distributed

	nb_filter1, nb_filter2, nb_filter3 = filters
	if K.image_dim_ordering() == 'tf':
		bn_axis = 3
	else:
		bn_axis = 1

	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = TimeDistributed(Convolution2D(nb_filter1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), input_shape=input_shape, name=conv_name_base + '2a')(input_tensor)
	x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = TimeDistributed(Convolution2D(nb_filter2, (1, kernel_size), padding='same', trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2b')(x)
	x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2c', trainable=trainable)(x)
	x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

	shortcut = TimeDistributed(Convolution2D(nb_filter3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '1')(input_tensor)
	shortcut = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '1')(shortcut)

	x = Add()([x, shortcut])
	x = Activation('relu')(x)
	return x


def classifier_layers(x, input_shape, stage_num, trainable=False):

	# compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
	# (hence a smaller stride in the region that follows the ROI pool)
	if K.backend() == 'tensorflow':
		x = conv_block_td(x, 3, [512, 512, 1024], stage=stage_num, block='a', input_shape=input_shape, strides=(1, 2), trainable=trainable)
	elif K.backend() == 'theano':
		x = conv_block_td(x, 3, [512, 512, 1024], stage=stage_num, block='a', input_shape=input_shape, strides=(1, 1), trainable=trainable)

	print 'INFO: Classifier layers x block a: ', x
	x = identity_block_td(x, 3, [512, 512, 1024], stage=stage_num, block='c', trainable=trainable)
	print 'INFO: Classifier layers x block b: ', x
	x = identity_block_td(x, 3, [512, 512, 1024], stage=stage_num, block='d', trainable=trainable)
	print 'INFO: Classifier layers x block c: ', x

	#x = TimeDistributed(AveragePooling2D((2, 1)), name='avg_pool')(x)

	return x



def classifier(base_layers, input_rois, num_rois, nb_classes = 2, trainable=False):

	# compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround

	if K.backend() == 'tensorflow':
		#pooling_regions = 14
		#input_shape = (num_rois, 1, 14, 3)
		pooling_regions = 14
		input_shape = (num_rois, 1, 14, 3)
	elif K.backend() == 'theano':
		pooling_regions = 2
		input_shape = (num_rois,2,2,24)
	#nb_classes = 2


	out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

	out = classifier_layers(out_roi_pool, input_shape=input_shape, stage_num = 5,  trainable=True)
	
	#input_shape = (num_rois, 1, 7, 1024)
	#out = classifier_layers(out, input_shape=input_shape, stage_num = 6, trainable=True)

	out = TimeDistributed(Flatten())(out)

	#out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
	out_class = Dense(nb_classes, activation='softmax', kernel_initializer='zero', name='dense_class_{}'.format(nb_classes))(out)
	print 'INFO: Output Classification',out_class
	# note: no regression target for bg class
	#out_regr = TimeDistributed(Dense(4 * (nb_classes), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
	out_regr = Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero', name='dense_regress_{}'.format(nb_classes))(out)
	#print 'INFO: Output Regression', out_regr
	return [out_class, out_regr]

