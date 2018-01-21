import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#使用numpy生成200个随机点
x_data = np.linspace(-0.5,0.5,200)[:, np.newaxis] #-0.5~0.5均匀分布200个点 []加维度变成二维 
noise = np.random.normal(0,0.02,x_data.shape) #生成一些干扰，形状和x_data形状一样
y_data = np.square(x_data) + noise

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,1]) #None:行可以任意 列为一列
y = tf.placeholder(tf.float32,[None,1])

#定义神经网络中间层
Weights_L1 = tf.Variable(tf.random_normal([1,10])) #形状一行十列
biases_L1 = tf.Variable(tf.zeros([1,10])) #初始化为零，十个偏置值
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1 #信号的总和 matmul:矩阵乘法
L1 = tf.nn.tanh(Wx_plus_b_L1) #用双曲正切函数作用于信号的总和（中间层的输出）

#定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([10,1]))
biases_L2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

#二次代价函数:（真实值-预测值）^2 / m m用于求平均
loss = tf.reduce_mean(tf.square(y-prediction))
#使用梯度下降法训练 学习率0.1，最小化损失
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    #变量初始化
    sess.run(tf.global_variables_initializer())
    #训练2000次 训练好模型
    for _ in range(2000):
        sess.run(train_step, feed_dict = {x:x_data, y:y_data}) #传入样本中的值
        
    #获得预测值
    prediction_value = sess.run(prediction,feed_dict={x:x_data})
    #画图
    plt.figure()
    plt.scatter(x_data,y_data)
    #r-:红色实线，径宽设为5
    plt.plot(x_data, prediction_value, 'r-', lw = 5)
    plt.show()
