"""
Fetch：
在op的一次运行中获取多个tensor值
使用Session对象的run()调用执行图时，传入一些tensor(以常量或变量的形式存储)，这些tensor会帮助你取回结果
"""
import tensorflow as tf
#Fetch  同时运行多个op
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input2, input3)
mul = tf.multiply(input1, add)
                  
with tf.Session() as sess:
    result = sess.run([mul, add]) #运行多个op
    print(result)
    
"""
Feed:
该机制可以临时替代图中的任意操作中的tensor
feed使用一个tensor值临时替代一个操作的输出结果
"""  
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    #Feed的数据以字典的形式传入
    print(sess.run(output,feed_dict = {input1:[8.], input2:[2.]}))
