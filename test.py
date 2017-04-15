# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
# 生成一个先入先出队列和一个QueueRunner
filenames = ['data_sample.txt']
filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
# 定义Reader
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
# 定义Decoder
example, label, matrix = tf.decode_csv(value, record_defaults=[['null'], ['null'],['null']])
# 运行Graph
with tf.Session() as sess:
    coord = tf.train.Coordinator()  #创建一个协调器，管理线程
    threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。
    for i in range(1):
        print( example.eval())
        print( label.eval()) 
        print( type(matrix.eval()))  #取样本的时候，一个Reader先从文件名队列中取出文件名，读出数据，Decoder解析后进入样本队列。
    coord.request_stop()
    coord.join(threads)
    # train_x = np.reshape(np.asarray(matrix.eval()),(101,101,60))
    # print(train_x)
    train_x = str(matrix.eval())
    train_x = train_x[2:-1]
    print(type(train_x))
    train_x = train_x.split()
    # print(train_x)
    train_x = np.reshape(np.asarray(train_x,dtype='float32'),(101,101,60))
    print(type(train_x))
    print(train_x.shape)
    print(train_x)