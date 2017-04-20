import keras
from keras.models import load_model
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
weight_init = 0.0002
dropout = 0.5
batch_size = 128
# iterations = 180
epochs = 180
log_filepath = r'./logs' + sys.argv[2] + '2'

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

if __name__ == '__main__':

	tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, histogram_freq=0)
	cbks = [tb_cb]

	# model = build_model()
	model = load_model(sys.argv[1])
	print("--------------generate_arrays_from_file--------------------")
	x, y = generate_arrays_from_file('train_B.txt')
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