import tensorflow as tf
import numpy as np

#使用numpy生成100个随机点
x_data = np.random.rand(100)
y_data = x_data*0.1 + 0.2

#构造一个线性模型
b = tf.Variable(0.) #表明是一个小数
k = tf.Variable(0.)
y = k*x_data + b

#二次代价函数
loss = tf.reduce_mean(tf.square(y_data-y)) #求平均值
#定义一个梯度下降法的优化器
optimizer = tf.train.GradientDescentOptimizer(0.2)
#定义一个最小化代价函数
train = optimizer.minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step%20 == 0:
            print(step,sess.run([k,b]))
                        
            
"""
0 [0.054392964, 0.10031929]
20 [0.10388039, 0.19788671]
40 [0.10224979, 0.1987748]
60 [0.10130439, 0.19928965]
80 [0.10075624, 0.19958816]
100 [0.10043845, 0.19976123]
120 [0.10025422, 0.19986156]
140 [0.10014738, 0.19991975]
160 [0.10008545, 0.19995347]
180 [0.10004954, 0.19997302]
200 [0.10002873, 0.19998436]
"""
