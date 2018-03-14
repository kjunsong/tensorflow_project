"""
tensorflow python库有一个默认图(default graph)
构建图的第一步是创建源op(source operation)，源op不需要任何输入，如 常量(Constant)
op构造器可以为其增加节点，op构造器的返回值代表被构造出的op的输出，这些返回值可以传递给其它op构造器
"""

import tensorflow as tf

#创建一个常量op(operation)，产生一个1*2矩阵（prnit(m1)：输出为向量而不是矩阵）。这个op被作为一个节点
#加到默认图中
#构造器的返回值代表该常量op的返回值
m1 = tf.constant([[3,3]])
#创建一个常量op
m2 = tf.constant([[2],[3]])

#创建一个矩阵乘法op，把m1，m2传入
product = tf.matmul(m1,m2)

#定义一个会话，启动默认图
sess = tf.Session()
#调用sess的run方法执行矩阵乘法op
#整个执行过程是自动化的，会话负责传递op所需的全部输入。op通常是并发执行的
#函数调用'run(product)'触发
result = sess.run(product)
print(result)

#任务完成，关闭会话
sess.close()

#也可以使用"with"代码块来自动完成关闭动作
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
    sess.close()
