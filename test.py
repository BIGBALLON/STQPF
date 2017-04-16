# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

BATCH_SIZE = 32
# 生成一个先入先出队列和一个QueueRunner
filenames = ['train.txt']
filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
# 定义Reader
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
# 定义Decoder
example, label, matrix = tf.decode_csv(value, record_defaults=[['null'], ['1.0'],['null']])
example_batch, label_batch, matrix_batch = tf.train.batch(  
      [example, label, matrix], batch_size=BATCH_SIZE)  
# 运行Graph
with tf.Session() as sess:
    coord = tf.train.Coordinator()  #创建一个协调器，管理线程
    threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。
    for i in range(1000):
        if coord.should_stop():
            break
        example, label, matrix = sess.run([example_batch,label_batch,matrix_batch])  

        x_train = np.asarray([])

        for i in range(BATCH_SIZE):
            tmp = str(matrix[i])
            tmp = tmp[2:-1]
            tmp = tmp.split()
            if i == 0 :
                x_train = np.reshape(np.asarray(tmp,dtype='float32'),(101,101,60))
            else:
                x_train =  np.concatenate((x_train,np.reshape(np.asarray(tmp,dtype='float32'),(101,101,60))), axis=0)
        x_train = np.reshape(x_train,(BATCH_SIZE,101,101,60))
        print(x_train.dtype)

    coord.request_stop()
    coord.join(threads)