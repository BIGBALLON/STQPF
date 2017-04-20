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
	return np.array(X)


model = load_model(sys.argv[1])
x = generate_arrays_from_file('testA.txt')
p = model.predict(x,128)

for l in p:
	print(l)