"""requirement: tensorflow version 1.x.x
"""
import numpy as np

from tframe import console
console.suppress_logging()

from tframe import tf

# -----------------------------------------------------------------------------
# Begin
# -----------------------------------------------------------------------------
"""
toy model: f(x) = a*x + b, in which a and b are unknown
given observations:
(x_1, y_1)
(x_2, y_2)
... ...
(x_N, y_N)
Find a, b
---------------
a = 2, b = 6
(1, 8)
(2, 10)
(3, 12)
(4, 14)
(5, 16)
"""
# Prepare toy data
N = 5
x = np.array(range(1, N + 1), dtype=np.float)
y = x * 2.0 + 6.0

#
init_value = [0.0, 0.0]
w = tf.Variable(initial_value=init_value, trainable=True, dtype=tf.float32)

# Initialize a session
sess = tf.Session()


# Construct loss tensor
# tensor_x, tensor_y = tf.constant(x, dtype=tf.float32), tf.constant(y, dtype=tf.float32)
tensor_x = tf.placeholder(dtype=tf.float32, shape=[N])
tensor_y = tf.placeholder(dtype=tf.float32, shape=[N])
feed_dict = {tensor_x: x, tensor_y: y}

X = tf.stack([x, tf.ones_like(x, dtype=tf.float32)], axis=-1)
Y = tf.reshape(tensor_y, [N, 1])
W = tf.reshape(w, [2, 1])

def loss_function(X, W, Y):
  # Forward pass
  D = tf.matmul(X, W) - Y
  return tf.reduce_sum(tf.abs(D))

loss = loss_function(X, W, Y)

# Construct an optimizer
optimizer = tf.train.GradientDescentOptimizer(0.001)
train_step = optimizer.minimize(loss)


# Initialize w in GPU
sess.run(tf.global_variables_initializer())

print('[0] loss = {}'.format(sess.run(loss, feed_dict)))
for t in range(99999999):
  sess.run(train_step, feed_dict)
  val_loss = sess.run(loss, feed_dict)
  print('[{}] loss = {}'.format(t + 1, val_loss))
  print('[a, b] = {}'.format(sess.run(w)))
  if val_loss < 0.1: break
