"""
MNIST数据集：
  MNIST数据集可在官网下载，这里用python源码自动下载和安装这个数据集
  数据集分为60000行的训练集(mnist.train)和10000行的测试集(mnist.test)。数据集中图片是mnist.train.images，标签是mnist.train.labels。
  每一张图片包含28*28像素，数组展成向量长度为784。展平图片丢失二维结构信息，但最优秀的计算机视觉方法会挖掘并利用这些结构信息。
Softmax回归：
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot = True)  #第一个参数为路径，one_hot把标签转化为只由0和1表示
#每个批次的大小：训练时候放入一个批次
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size   # //为整除，总的数量整除批次

"""实现回归模型"""
#定义两个placeholder（占位符），用2维的浮点张量来表示这些图，这个张量形状是[None,784]。(这里的None表示此张量的第一个维度可以是任何长度)
#在tensorflow运行计算时输入这个值，我们希望能够输入任意变量的MNIST图像，每一张图展平成784维的向量。
x = tf.placeholder(tf.float32,[None, 784])  #当传入参数的时候，None->100(批次大小)
y = tf.placeholder(tf.float32,[None, 10])
#一个Variable代表一个可修改的张量，在这里用全为零的张量来初始化w和b
W = tf.Variable(tf.zeros([784,10])) #w的维度是[784,10]因为我们想要用784维的图片向量乘以它，得到一个10维的证据值向量，每一对对应不同数字类。
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b) #返回的是概率，tf.matmul(x,W)表示x乘以W

"""训练模型"""
#需要定义一个指标来评估这个模型好坏，用一个指标称为成本(cost)或者损失(loss)，在此使用二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
#使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#已经设置好了模型。运算之前，添加一个操作来初始化我们创建的变量
init = tf.global_variables_initializer()

"""评估我们的模型"""
# tf.equal 比较变量 标签一样返回true，不一样返回false，结果存放在一个bool型列表中
# tf.argmax 返回概率最大的那个值的位置，相当于数字标签
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
#求准确率 tf.reduce_mean求平均值 tf.cast将bool类型转化为tf.float32类型，true->1.0 false->0
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #例如：[True,False,True,True]会变成[1,0,1,1],取平均得到0.75
#在一个Session里启动我们的模型，并初始化变量
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21): #迭代21个周期
        for batch in range(n_batch): #每个周期中批次数
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) #获得批次 batch_xs保存数据 batch_ys保存标签
            sess.run(train_step, feed_dict = {x:batch_xs, y:batch_ys}) #进行一次训练
        
        #求准确率
        #mnist.test.images测试集图片 mnist.test.labels测试集标签
        acc = sess.run(accuracy, feed_dict = {x:mnist.test.images, y:mnist.test.labels})
        print("Iter" + str(epoch)+",Testing Accuracy" + str(acc))

#最终结果大约是91%，并不好，需要改进        
"""
Extracting MNIST_data\train-images-idx3-ubyte.gz
Extracting MNIST_data\train-labels-idx1-ubyte.gz
Extracting MNIST_data\t10k-images-idx3-ubyte.gz
Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
Iter0,Testing Accuracy0.8306
Iter1,Testing Accuracy0.8709
Iter2,Testing Accuracy0.8819
Iter3,Testing Accuracy0.8883
Iter4,Testing Accuracy0.8947
Iter5,Testing Accuracy0.8972
Iter6,Testing Accuracy0.9002
Iter7,Testing Accuracy0.9012
Iter8,Testing Accuracy0.904
Iter9,Testing Accuracy0.905
Iter10,Testing Accuracy0.9064
Iter11,Testing Accuracy0.9073
Iter12,Testing Accuracy0.9087
Iter13,Testing Accuracy0.9095
Iter14,Testing Accuracy0.9096
Iter15,Testing Accuracy0.911
Iter16,Testing Accuracy0.9113
Iter17,Testing Accuracy0.9122
Iter18,Testing Accuracy0.9138
Iter19,Testing Accuracy0.9136
Iter20,Testing Accuracy0.9135
"""
