import keras
import tensorflow as tf 
import numpy as np
import math
import os 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.initializers import RandomNormal  
from keras import optimizers
from keras.layers.normalization import BatchNormalization

acti_fun = 'relu'
weight_init = 0.0001
dropout = 0.5
batch_size = 50
iterations = 180
epochs = 50
log_filepath = r'./logs'

def get_he_weight(layer,k,c):
	return math.sqrt(2/(k*k*c))

def process_line(line):
	tmp = [ val for val in line.strip().split(',')]
	y = np.array(tmp[1],dtype='float32')
	x = str(tmp[2])
	x = x.split()
	x = np.reshape(np.asarray(x,dtype='float32'),(101,101,60))
	x = x[45:65,45:65,:]
	return x,y

def generate_arrays_from_file(path,batch_size):
	if not os.path.exists(path):
		print("file not exist")
		return
	while 1:
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
			cnt += 1
			if cnt==batch_size:
				cnt = 0
				yield (np.array(X), np.array(Y))
				X = []
				Y = []
	f.close()

#  model should be change
def build_model():

	model = Sequential()

	model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(1,5,192)), input_shape=(20,20,60)))
	model.add(BatchNormalization())  
	model.add(Activation(acti_fun))

	model.add(Conv2D(160, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(2,1,160))))
	model.add(BatchNormalization())
	model.add(Activation(acti_fun))

	model.add(Conv2D(96, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,1,96)) ))
	model.add(BatchNormalization())  
	model.add(Activation(acti_fun))

	model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'same'))

	model.add(Dropout(dropout))

	model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(4,5,192))))
	model.add(BatchNormalization())
	model.add(Activation(acti_fun))

	model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(5,1,192))))
	model.add(BatchNormalization()) 
	model.add(Activation(acti_fun))

	model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(6,1,192))))
	model.add(BatchNormalization())  
	model.add(Activation(acti_fun))

	model.add(AveragePooling2D(pool_size=(3, 3),strides=(2,2),padding = 'same'))

	model.add(Dropout(dropout))
	model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(7,3,192))))
	model.add(BatchNormalization())  
	model.add(Activation(acti_fun))

	model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(8,1,192))))
	model.add(BatchNormalization())  
	model.add(Activation(acti_fun))

	model.add(Conv2D(10, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(9,1,10))))
	model.add(BatchNormalization())  
	model.add(Activation(acti_fun))


	model.add(Flatten())
	model.add(Dense(1))

	# optimizers should be tested
	# sgd + momentum
	# others
	adam = optimizers.Adam(lr=0.0035, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
	# sgd = optimizers.SGD(lr=0.005, momentum=0.9, decay=1e-6, nesterov=True)
	# rms = optimizers.RMSprop(lr=0.0035, rho=0.9, epsilon=1e-08, decay=1e-6)
	model.compile(optimizer=adam, loss='mse')
	return model

if __name__ == '__main__':

	tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, histogram_freq=0)
	cbks = [tb_cb]

	model = build_model()
	model.fit_generator(
			generator = generate_arrays_from_file(r'train_split.txt',batch_size=batch_size),
			samples_per_epoch=iterations,
			nb_epoch=epochs,
			callbacks=cbks,
			validation_data=generate_arrays_from_file(r'test_split.txt',batch_size=batch_size),
			validation_steps=15,
			max_q_size=50, 
			workers=1)
	model.save('test1.h5')