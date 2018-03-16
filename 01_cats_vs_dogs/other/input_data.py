# -*- coding: utf-8 -*-

"""
The aim of this project is to use TensorFlow to process our own data
   - input_data.py: read in data and generate batches 
   - model: build the model architecture
   - training: train

I used Windows10 with anaconda3.6, TensorFlow1.4
With current setting, 10000 training steps needed x minutes on my laptop.

data: cats vs. dogs from Kaggle
Download link: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
date size: test is 271M and train is 543M

How to run?
1. run the training.py once
2. call the run_training() in the console to train the model

Note:
    it is suggusted to restart your kenel to train the model multiple times(clear all the variables in the memory)
    Otherwise errors may occur:conv1/Weights/biases already exist......

"""

#%%
import tensorflow as tf
import numpy as np
import os

#%%
#the train file directory
#file name: cat.0.jpg
train_dir = 'C:/Users/俊松/Documents/我的工作区/My-TensorFlow-tutorials-master/01.cats_vs_dogs/data/train/'   
#%%
def get_files(file_dir):
    '''
    Args:
        file_dir:file directory
    Returns:
        list of images and lables
    '''
    cats = []
    label_cats = []
    dogs =[]
    label_dogs = []
    for file in os.listdir(file_dir):       #返回这个路径下所有文件的名字
        name = file.split(sep='.')          #以'.'分离file（separate） 
        if name[0] == 'cat':
            cats.append(file_dir + file)    #文件路径
            label_cats.append(0) 
            #print(file_dir)
            #print(file)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
                                            # 12500 cats & 12500 dogs
    print('There are %d cats\nThere are %d dogs'%(len(cats), len(dogs))) #最后一个%是输出语法
    
    image_list = np.hstack((cats, dogs))    # np.hstack() 按顺序堆叠起来
    label_list = np.hstack((label_cats, label_dogs)) 
    temp = np.array([image_list, label_list])  #np.array() 创建一个矩阵，temp[0]是图片路径，temp[1]第二列是标签
    """
    print(image_list[0]) : C:/Users.../cat.0.jpg
    print(label_list[0]) : 0
    print(temp[1])       : ['0' '0' '0' ..., '1' '1' '1']
    print(np.array([[2,3],[2,3]]) 
                         : [[2 3]
                            [2 3]]
    
    """
    temp = temp.transpose()                 #相当于将矩阵进行转置 变成两行，25000列矩阵
    np.random.shuffle(temp)                 #将图片打乱
    
    image_list = list(temp[:, 0])           #list:(元组)转化成列表(元组放在括号中，列表是放于方括号中)
    label_list = list(temp[:, 1])           #后面的 0,1是什么意思
    label_list = [int(i) for i in label_list] #将字符'1' 转化成 1
    
    return image_list, label_list

#%%
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue   容量：队列中最大元素
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''  
    image = tf.cast(image, tf.string)       #转换数据类型
    label = tf.cast(label, tf.int32)

    # make an input queue 输入队列
    input_queue = tf.train.slice_input_producer([image, label]) #可用来每次产生切片 ？？？
  
    label = input_queue[1]                  #标签
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3) #解码
    ######################################
    # data argumentation should go to here 多做一些数据的特征加强
    ######################################
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H) #裁剪图片的长宽（图像需要3或4个维度）
    # if you want to test the generated batches of images, you might want to comment the following line.
    # 如果想看到正常的图片，请注释掉行（标准化）和 行（image_batch = tf.cast(image_batch, tf.float32)）
    # 训练时不要注释掉！
    image = tf.image.per_image_standardization(image) #？？？神经网络对图片要求很高，所以标准化图片减去均值除以方差
    
    image_batch, label_batch = tf.train.batch([image, label],   
                                                batch_size= batch_size,
                                                num_threads= 8, #电脑线程数
                                                capacity = capacity) #队列中最多能容纳多少个元素 
    
    #you can also use shuffle_batch 数据会打乱 因为我们已经打乱了数据 所以用上面的函数
#    image_batch, label_batch = tf.train.shuffle_batch([image,label],  
#                                                      batch_size=BATCH_SIZE,
#                                                      num_threads=64,
#                                                      capacity=CAPACITY,
#                                                      min_after_dequeue=CAPACITY-1)
    
    label_batch = tf.reshape(label_batch, [batch_size])        #将tensor变换为参数shape形式？？？
    image_batch = tf.cast(image_batch, tf.float32)             #对输入的tensor进行数据转换 
    
    return image_batch, label_batch

#%%
# To test the generated batches of images
# When training the model, DO comment the following codes
''' 
import matplotlib.pyplot as plt

BATCH_SIZE = 8
CAPACITY = 256  #？？？   队列中最多能容纳多少个元素 
IMG_W = 208
IMG_H = 208

#train_dir = 'C:/Users/俊松/Documents/我的工作区/My-TensorFlow-tutorials-master/1 cats vs dogs/cats_vs_dogs/data/train/'

image_list, label_list = get_files(train_dir)
image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

                                            #结束后 with会自动关闭打开的文件
with tf.Session() as sess:                  #会话
    i = 0
    print(0)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord) #每次读到什么地方，下次从哪读Coordinator：协调 queue（读Q）：队列
    
    try:
        while not coord.should_stop() and i<1:
            print (1)
            img, label = sess.run([image_batch, label_batch])
            #just test one batch
            for j in np.arange(BATCH_SIZE):
                print(2)
                print('label: %d' %label[j])
                plt.imshow(img[j,:,:,:])  #4D tensor
                plt.show()
            i += 1
        
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)
'''  
#%%






