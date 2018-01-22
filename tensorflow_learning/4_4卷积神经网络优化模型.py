#传统网络能达到98%的准确率，使用卷积神经网络能达到99%的准确率
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
batch_size = 100
n_batch = mnist.train.num_examples // batch_size

"""
构建一个多层卷积网络
为了创建这个模型，我们需要创建大量的权值和偏置项。这个模型中的权重在初始化时应该加入少量的噪声来打破对称性以及避免0梯度。
我们使用ReLU神经元，因此比较好的做法是一个较小的正数来初始化偏置项，避免神经元节点输出恒为0的问题（dead neurons）。
"""
#初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) #生成一个截断的正态分布
    return tf.Variable(initial)
#初始化偏振
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

#卷积层
def conv2d(x,W):
    #2d:二维
    #x input tensor of shape [batch, in_height, in_width, in_channels]
    #W filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
    #strides[0]=strides[3]=1 strides[1]代表x方向的步长，strides[2]代表y方向的步长
    #padding:"SAME","VALID" padding方式有两种，一种是SAME一种VALID
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
#池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,784]) #28*28 转为 一维向量
y = tf.placeholder(tf.float32,[None,10])

#改变x的格式转为4D的向量[batch:批次, in_height, in_width, in_channels:灰白图像]
x_image = tf.reshape(x,[-1,28,28,1])

#初始化第一层卷积层的权值和偏振
W_conv1 = weight_variable([5,5,1,32]) #5*5的采样窗口，32个卷积核（32个特征平面）从1个平面抽取特征
b_conv1 = bias_variable([32]) #每一个卷积核一个偏置值
#为了用这一层，我们把x变成一个4d向量，2、3维对应图片的宽和高，最后一维代表图像的通道数，这里是灰度图。
#x_image = tf.reshape(x,[-1,28,28,1])

#把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数，最后进行max_pooling
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1) #进行max_pooling

#初始化第二个卷积层的权值和偏置
W_conv2 = weight_variable([5,5,32,64])#5*5的采样窗口，64个卷积核从32个平面抽取特征
b_conv2 = bias_variable([64]) #每一个卷积核一个偏置值
#把h_pool1和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) #进行max_pooling

#28*28的图片第一次卷积后还是28*28，第一次池化后变成14*14
#第二次卷积后为14*14，第二次池化后为7*7
#进行上面操作后得到64张7*7的平面

"""
密集连接层
初始化第一个全连接层的权值
把池化层2的输出的张量reshape为1维，乘上权值矩阵，加上偏置，然后对其使用ReLU
"""
W_fc1 = weight_variable([7*7*64, 1024]) #上一层有7*7*64个神经元，全连接层有1024个神经元，用于处理整个图片
b_fc1 = bias_variable([1024]) #1024个节点

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64]) #-1代表批次任意值，这里的批次是100
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1) #求第一个全连接层的输出

#Dropout    keep_prob用来表示神经元的输出概率
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

"""输出层"""
#初始化第二个全连接层
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2) #计算输出(转化为概率)

"""训练和评估模型"""
#交叉熵代价函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#结果存放在一个布尔列表中
correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1)) #argmax返回一维张量中最大的值所在的位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):   #训练21个周期
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) #传入一个批次的数据
            sess.run(train_step, feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7}) #喂到placeholder里面 
            
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
        print("Iter"+str(epoch)+",Testing Accuracy="+str(acc))
