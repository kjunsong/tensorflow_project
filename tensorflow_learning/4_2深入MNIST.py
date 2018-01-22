"""

"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集，第一个参数为路径，one_hot把标签转化为只有0和1
mnist = input_data.read_data_sets("MNIST_data",one_hot = True)
batch_size = 100
n_batch = mnist.train.num_examples // batch_size 

x = tf.placeholder(tf.float32,[None, 784])
y = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32) #目的：加中间层

W1 = tf.Variable(tf.truncated_normal([784,2000],stddev=0.1))
b1 = tf.Variable(tf.zeros([2000]+0.1))
L1 = tf.nn.tanh(tf.matmul(x,W1)+b1)
L1_drop = tf.nn.dropout(L1,keep_prob)

loss  = tf.reduce_mean(tf.square(y-prediction))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
init = tf.global_varibales_initialzer()

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
