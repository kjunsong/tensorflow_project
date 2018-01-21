import tensorflow as tf

#创建一个变量 初始化为标量0 变量名为counter
state = tf.Variable(0, name = 'counter') 
#创建一个op，作用是使state加1
one = tf.constant(1)
new_value = tf.add(state,one)
update = tf.assign(state, new_value) #赋值op

#启动图后，变量必须经过初始化(init)op
#增加一个初始化op到图中
init = tf.global_variables_initializer()

#启动图 运行op
with tf.Session() as sess:
    #运行'init'op
    sess.run(init)
    #打印'start'的初始值
    print(sess.run(state))
    #运行op 更新并打印'start'
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))     
