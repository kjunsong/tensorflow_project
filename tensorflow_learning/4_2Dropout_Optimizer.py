"""
Dropout(中断)：为了减少过拟合，在输出层之前加入dropout
使用的模型比之前的复杂，训练时间较长。五万张图片属于少量的图片，用来训练特别复杂的模型，如googlenet，会出现过拟合现象。
Optimizer(优化器)：
各种优化器对比
标准梯度下降法：数据量大的时候使用，但是每次更新权值较慢(tf.train.GradientDescentOptimizer)
    先计算所有样本汇总误差，然后根据总误差来更新权值
随机梯度下降法：更新权值速度较快，但是会引入噪声
    随机抽取一个样本来计算误差，然后更新权值
批量梯度下降法：是一个折中的方案
    从总样本中选取一个批次（一共有10000个样本，随机选取100个样本作为一个batch），然后计算这个batch的误差，根据总误差来更新权值。
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data",one_hot = True)  #载入数据集，第一个参数为路径，one_hot把标签转化为只有0和1
batch_size = 100
n_batch = mnist.train.num_examples // batch_size 

"""占位符"""
x = tf.placeholder(tf.float32,[None, 784]) #输入图片x是一个2维的浮点数张量，这里分配它的shape为[None,784],其中784是一张展平的MNIST图片维度
y = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32) #目的：加中间层

"""变量"""
#一个变量代表着tensorflow计算图中的一个值，能够在计算过程中使用，甚至进行修改
W1 = tf.Variable(tf.truncated_normal([784,2000],stddev=0.1)) #创建一个简单的神经网络 权值W可以使用tf.truncated_normal优化
b1 = tf.Variable(tf.zeros([2000])+0.1)
L1 = tf.nn.tanh(tf.matmul(x,W1)+b1)
L1_drop = tf.nn.dropout(L1,keep_prob) 

W2 = tf.Variable(tf.truncated_normal([2000,2000],stddev=0.1))
b2 = tf.Variable(tf.zeros([2000])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop,W2)+b2)
L2_drop = tf.nn.dropout(L2,keep_prob) 

W3 = tf.Variable(tf.truncated_normal([2000,1000],stddev=0.1))
b3 = tf.Variable(tf.zeros([1000])+0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop,W3)+b3)
L3_drop = tf.nn.dropout(L3,keep_prob) 

W4 = tf.Variable(tf.truncated_normal([1000,10],stddev=0.1))
b4 = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.nn.softmax(tf.matmul(L3_drop,W4)+b4)

loss  = tf.reduce_mean(tf.square(y-prediction))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss) #不同优化器效果不同
init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(init)
  for epoch in range(31):
    for batch in range(n_batch):
      batch_xs, batch_ys = mnist.train.next_batch(batch_size)
      sess.run(train_step, feed_dict = {x:batch_xs, y:batch_ys, keep_prob:1.0})
      
    test_acc = sess.run(accuracy, feed_dict = {x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
    train_acc = sess.run(accuracy, feed_dict = {x:mnist.train.images, y:mnist.train.labels, keep_prob:1.0})
    print("Iter"+str(epoch)+",Testing Accuracy"+str(test_acc)+"Training Accuracy"+str(train_acc))
   
"""
Extracting MNIST_data\train-images-idx3-ubyte.gz
Extracting MNIST_data\train-labels-idx1-ubyte.gz
Extracting MNIST_data\t10k-images-idx3-ubyte.gz
Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
Iter0,Testing Accuracy0.919Training Accuracy0.928909
Iter1,Testing Accuracy0.9368Training Accuracy0.956109
...
"""
