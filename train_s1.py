import keras
import tensorflow as tf 
import numpy as np
import math
import os 
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.initializers import RandomNormal  
from keras import optimizers
from keras.layers.normalization import BatchNormalization

acti_fun = 'relu'
weight_init = 0.0003
dropout = 0.5
batch_size = 128
# iterations = 180
epochs = 200
log_filepath = r'./logs'

def get_he_weight(k,c):
	return math.sqrt(2/(k*k*c))

def process_line(line):
	tmp = [ val for val in line.strip().split(',')]
	y = np.array(tmp[1],dtype='float32')
	x = str(tmp[2])
	x = x.split()
	x = np.reshape(np.asarray(x,dtype='float32'),(101,101,60))
	x = x[40:70,40:70,:]
	return x, y

def generate_arrays_from_file(path):
	if not os.path.exists(path):
		print("file not exist")
		return
	f = open(path)
	cnt = 0
	X =[]
	Y =[]
	for line in f:
		ll = len(line)
		if ll < 100:
			continue
		x, y = process_line(line)
		X.append(x)
		Y.append(y)
	f.close()
	return np.array(X), np.array(Y)

#  model should be change
#  model should be change
def build_model():

	# -------- build model -------- #
	model = Sequential()
	# Block 1
	model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,64)), name='block1_conv1', input_shape=(30,30,60)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,64)), name='block1_conv2'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

	# Block 2
	model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,128)), name='block2_conv1'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,128)), name='block2_conv2'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

	# Block 3
	model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,256)), name='block3_conv1'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,256)), name='block3_conv2'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,256)), name='block3_conv3'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,256)), name='block3_conv4'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

	# Block 4
	model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,512)), name='block4_conv1'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,512)), name='block4_conv2'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,512)), name='block4_conv3'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,512)), name='block4_conv4'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

	# Block 5
	model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(get_he_weight(3,512)), name='block5_conv1'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(get_he_weight(3,512)), name='block5_conv2'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(get_he_weight(3,512)), name='block5_conv3'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(get_he_weight(3,512)), name='block5_conv4'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	# model modification for cifar-10
	model.add(Flatten(name='flatten'))
	model.add(Dense(4096, use_bias = True, kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = 0.01), name='fc_cifa10'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(dropout))
	model.add(Dense(4096, kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = 0.01), name='fc2'))  
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(dropout))      
	model.add(Dense(1, kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = 0.01), name='predictions_cifa10'))        
	# model.add(Flatten())

	# optimizers should be tested
	# sgd + momentum
	# others
	adam = optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
	# sgd = optimizers.SGD(lr=0.05, momentum=0.9, decay=1e-6, nesterov=True)
	# rms = optimizers.RMSprop(lr=0.0035, rho=0.9, epsilon=1e-08, decay=1e-6)
	model.compile(optimizer=adam, loss='mse')
	return model

if __name__ == '__main__':

	# tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, histogram_freq=0)
	# cbks = [tb_cb]

	model = build_model()

	print("--------------generate_arrays_from_file--------------------")
	x, y = generate_arrays_from_file('train_A.txt')
	print("----------------------------ok-----------------------------")
	
	model.fit( x, y, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=None, validation_split=0.1, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)


	# model.fit_generator(
	# 		generator = generate_arrays_from_file(r'train_split.txt',batch_size=batch_size),
	# 		samples_per_epoch=iterations,
	# 		nb_epoch=epochs,
	# 		callbacks=cbks,
	# 		validation_data=generate_arrays_from_file(r'test_split.txt',batch_size=batch_size),
	# 		validation_steps=20,
	# 		workers=1)
	model.save(sys.argv[1])