import tensorflow as tf
import numpy as np
import roi_pooling_op


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

array = np.random.rand(32, 100, 100, 3)
data = tf.convert_to_tensor(array, dtype=tf.float32)
rois = tf.convert_to_tensor([[0, 10, 10, 20, 20], [1, 30, 30, 40, 40]], dtype=tf.float32)

W = weight_variable([3, 3, 3, 3])
h = conv2d(data, W)

[y, argmax] = roi_pooling_op.roi_pool(h, rois, 6, 6, 1.0/3)
y_data = tf.convert_to_tensor(np.ones((2, 6, 6, 1)), dtype=tf.float32)

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

# Launch the graph.
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sess = tf.Session()
sess.run(init)
for step in range(1):
    print(step)
    sess.run(train)
    a = sess.run(y)
    print(a.shape)

