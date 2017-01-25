# -*- coding: utf-8 -*-
import tensorflow as tf
from mnist import read_data_sets

input_data = read_data_sets('MNIST_data', one_hot=True)

# x = tf.placeholder("float", [None, 784])
x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

# y_ = tf.placeholder("float", [None,10])
y_ = tf.placeholder(tf.float32, [None, 10])

# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
# init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)


for i in range(1000):
    batch_xs, batch_ys = input_data.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print sess.run(accuracy, feed_dict={x: input_data.test.images, y_: input_data.test.labels})
