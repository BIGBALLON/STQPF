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
weight_init = 0.0005
dropout = 0.5
batch_size = 32
# iterations = 180
epochs = 180
log_filepath = r'./logs' + sys.argv[2]

lr_set = int(sys.argv[2])
print(lr_set)
print(type(lr_set))

if lr_set == 3:
	init_lr = 0.01
elif lr_set == 4:
	init_lr = 0.02
elif lr_set == 5:
	init_lr = 0.002
elif lr_set == 6:
	init_lr = 0.001
elif lr_set == 7:
	init_lr = 0.003
elif lr_set == 8:
  init_lr = 0.0008
else:
	init_lr = 0.005


print(init_lr)
print(log_filepath)


def get_he_weight(k,c):
	return math.sqrt(2/(k*k*c))

def process_line(line):
	tmp = [ val for val in line.strip().split(',')]
	y = np.array(tmp[1],dtype='float32')
	x = str(tmp[2])
	x = x.split()
	x = np.reshape(np.asarray(x,dtype='float32'),(101,101,60))
	# x = x[:,:,:]
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

	model.add(Conv2D(160, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(5,160)), input_shape=(101,101,60)))
	model.add(BatchNormalization())  
	model.add(Activation(acti_fun))

	model.add(Conv2D(160, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(1,160))))
	model.add(BatchNormalization())
	model.add(Activation(acti_fun))

	model.add(Conv2D(96, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(1,96)) ))
	model.add(BatchNormalization())  
	model.add(Activation(acti_fun))

	model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'same'))

	model.add(Dropout(dropout))

	model.add(Conv2D(96, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(5,96))))
	model.add(BatchNormalization())
	model.add(Activation(acti_fun))

	model.add(Conv2D(96, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(1,96))))
	model.add(BatchNormalization()) 
	model.add(Activation(acti_fun))

	model.add(Conv2D(96, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(1,96))))
	model.add(BatchNormalization())  
	model.add(Activation(acti_fun))

	model.add(AveragePooling2D(pool_size=(3, 3),strides=(2,2),padding = 'same'))

	model.add(Dropout(dropout))
	model.add(Conv2D(96, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,96))))
	model.add(BatchNormalization())  
	model.add(Activation(acti_fun))

	model.add(Conv2D(96, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(1,96))))
	model.add(BatchNormalization())  
	model.add(Activation(acti_fun))

	model.add(Conv2D(10, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(1,10))))
	model.add(BatchNormalization())  
	model.add(Activation(acti_fun))

	model.add(Flatten())
	model.add(Dense(1))

	# optimizers should be tested
	# sgd + momentum
	# others
	adam = optimizers.Adam(lr=init_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
	# sgd = optimizers.SGD(lr=0.05, momentum=0.9, decay=1e-6, nesterov=True)
	# rms = optimizers.RMSprop(lr=0.0035, rho=0.9, epsilon=1e-08, decay=1e-6)
	model.compile(optimizer=adam, loss='mse')
	return model

if __name__ == '__main__':

	tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, histogram_freq=0)
	cbks = [tb_cb]

	model = build_model()

	print("--------------generate_arrays_from_file--------------------")
	x, y = generate_arrays_from_file('train_A.txt')
	print("----------------------------ok-----------------------------")

	model.fit( x, y, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=cbks, validation_split=0.1, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)


	# model.fit_generator(
	# 		generator = generate_arrays_from_file(r'train_split.txt',batch_size=batch_size),
	# 		samples_per_epoch=iterations,
	# 		nb_epoch=epochs,
	# 		callbacks=cbks,
	# 		validation_data=generate_arrays_from_file(r'test_split.txt',batch_size=batch_size),
	# 		validation_steps=20,
	# 		workers=1)
	model.save(sys.argv[1])