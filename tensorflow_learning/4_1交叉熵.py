import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNISTZ_data",one_hot = True)
batch_size = 100
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32,[None, 784])
y = tf.placeholder(tf.float32,[None,10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

#二次代价函数 loss = tf.reduce_mean(tf.square(y-prediction))
#使用交叉熵定义代价函数，加速模型收敛速度
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
  sess.run(init)
  for epoch in range(21):
    for batch in range(n_batch): 
      batch_xs, batch_ys = mnist.train.next_batch(batch_size) #batch_xs保存数据 batch_ys保存标签
      sess.run(train_step,feed_dict = {x:batch_xs, y:batch_ys}) #进行一次训练
      
    acc = sess.run(accuracy, feed_dict = {x:mnist.test.images, y:mnist.test.labels})
    print("Iter"+str(epoch)+",Testing Accuracy"+str(acc))
    
"""
Extracting MNISTZ_data\train-images-idx3-ubyte.gz
Extracting MNISTZ_data\train-labels-idx1-ubyte.gz
Extracting MNISTZ_data\t10k-images-idx3-ubyte.gz
Extracting MNISTZ_data\t10k-labels-idx1-ubyte.gz
Iter0,Testing Accuracy0.8255
Iter1,Testing Accuracy0.8816
Iter2,Testing Accuracy0.9006
Iter3,Testing Accuracy0.9048
Iter4,Testing Accuracy0.9085
Iter5,Testing Accuracy0.9107
Iter6,Testing Accuracy0.9119
Iter7,Testing Accuracy0.9126
Iter8,Testing Accuracy0.9143
Iter9,Testing Accuracy0.916
Iter10,Testing Accuracy0.9169
Iter11,Testing Accuracy0.918
Iter12,Testing Accuracy0.9194
Iter13,Testing Accuracy0.9197
Iter14,Testing Accuracy0.9196
Iter15,Testing Accuracy0.9192
Iter16,Testing Accuracy0.9206
Iter17,Testing Accuracy0.9209
Iter18,Testing Accuracy0.9209
Iter19,Testing Accuracy0.9213
Iter20,Testing Accuracy0.9217
"""

