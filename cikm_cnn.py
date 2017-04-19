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

acti_fun = 'relu'
weight_init = 0.00015
dropout = 0.5
batch_size = 100
iterations = 90
epochs = 100

def get_he_weight(layer,k,c):
    return math.sqrt(2/(k*k*c))

def process_line(line):
    tmp = [ val for val in line.strip().split(',')]
    y = np.array(tmp[1],dtype='float32')
    x = str(tmp[2])
    x = x.split()
    x = np.reshape(np.asarray(x,dtype='float32'),(101,101,60))
    x = x[50:60,50:60,1:4]
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

def build_model():

    model = Sequential()
  
    model.add(Conv2D(60, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(1,3,60)), input_shape=(10,10,3))) 
    model.add(Activation(acti_fun))
    model.add(Conv2D(60, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,1,60))))
    model.add(Activation(acti_fun))
    model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'same'))
    model.add(Dropout(dropout))

    model.add(Conv2D(60, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(4,3,60))))
    model.add(Activation(acti_fun))
    model.add(AveragePooling2D(pool_size=(3, 3),strides=(2,2),padding = 'same'))
    model.add(Dropout(dropout))

    model.add(Conv2D(60, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(8,1,60))))
    model.add(Activation(acti_fun))
    model.add(Conv2D(10, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(9,1,10))))
    model.add(Activation(acti_fun))

    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse')
    return model

if __name__ == '__main__':

    model = build_model()
    model.fit_generator(
            generator = generate_arrays_from_file(r'train_split.txt',batch_size=batch_size),
            samples_per_epoch=iterations,
            nb_epoch=epochs,
            validation_data=generate_arrays_from_file(r'test_split.txt',batch_size=batch_size),
            validation_steps=iterations)