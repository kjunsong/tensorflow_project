import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNISTZ_data",one_hot = True)
batch_size = 100
n_batch = mnist_train.num_examples // batch_size

x = tf.placeholder(tf.float32,[None, 784])
y = tf.placeholder(tf.float32,[None,10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

#二次代价函数 loss = tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.sotfmax_cross_entropy_with_logits(labels=y,logits=prediction))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y,q), tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
  sess.run(init)
  for epoch in range(21):
    for batch in range(n_batch): 
      batch_xs, batch_ys = mnist.train.next_batch(batch_size) #batch_xs保存数据 batch_ys保存标签
      sess.run(train_step,feed_dict = {x:batch_xs, y:batch_ys}) #进行一次训练
      
    acc = sess.run(accuracy, feed_dict = {x:mnist.test.images, y:mnist.test.labels})
    print("Iter"+str(epoch)+",Testing Accuracy"+str(acc))

