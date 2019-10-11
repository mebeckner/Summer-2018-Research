import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

#Using capped gradients and vars, softmax crossentropy with logits, relu instead of sigmoid

import tensorflow as tf
import numpy as np
from pylab import *
tf.logging.set_verbosity(tf.logging.ERROR)
#%%

tf.set_random_seed(123) #initial point from which to start optimizing- repeatable results
np.random.seed(123)

# Network parameters
num_input = 784 # Number of neurons in input layer- 28 x 28 pixels
num_hidden1 = 500 # Number of neurons in hidden layer 1
num_hidden2 = 150 # Number of neurons in hidden layer 2
num_output = 10 # Number of neurons in output layer- 10 possible digits
iteration = 1000 # Number of iterations in training step 1 2 3
batch_size = 100 # Number of inputs in one batch
learning_rate1 = 10
learning_rate2 = 10
learning_rate3 = 0.5 
beta = 5e-4 # weight for sparsity penalty

#%%
# Tensorflow graph input
x = tf.placeholder(tf.float32, [None, num_input])  # Input layer: placeholder to assign data to later
y_ = tf.placeholder(tf.float32, [None, num_output])  # True values for output layer

#%%

# Weights
W1_en = tf.Variable(tf.truncated_normal([num_input, num_hidden1], stddev = 0.1, seed = 123))
W2_en = tf.Variable(tf.truncated_normal([num_hidden1, num_hidden2], stddev = 0.1, seed = 123))
W3 = tf.Variable(tf.truncated_normal([num_hidden2, num_output], stddev=0.1, seed = 123))
#W1_de = tf.transpose(W1_en) # decoder step
#W2_de = tf.transpose(W2_en) # decoder step

# Biases
b1_en = tf.Variable(tf.zeros([num_hidden1]))
b2_en = tf.Variable(tf.zeros([num_hidden2]))
b3 = tf.Variable(tf.zeros([num_output]))
b1_de = tf.Variable(tf.zeros([num_input])) #a decoder step
b2_de = tf.Variable(tf.zeros([num_hidden1])) #a decoder step

#%%

# Hidden layers with ReLu activation
h1 = tf.nn.relu(tf.add(tf.matmul(x, W1_en), b1_en))
h2 = tf.nn.relu(tf.add(tf.matmul(h1, W2_en), b2_en))
y = tf.matmul(h2, W3) + b3

# Define constants
rho = tf.constant(0.05, shape = [1, num_hidden1])
rho0 = tf.constant(0.05, shape = [1, num_hidden2])

#%%
#kl_div_loss1 = tf.contrib.distributions.kl(rho, tf.reduce_mean(h1,1))
kl_div_loss1 = tf.reduce_sum(-tf.nn.softmax_cross_entropy_with_logits(labels = rho, logits = rho/tf.reduce_mean(h1,0)))
loss1 = tf.reduce_mean(tf.square(tf.nn.sigmoid(tf.matmul(h1, tf.transpose(W1_en))+b1_de) - x)) + beta * kl_div_loss1

#kl_div_loss2 = tf.contrib.distributions.kl(rho, tf.reduce_mean(h2,1))
kl_div_loss2 = tf.reduce_sum(-tf.nn.softmax_cross_entropy_with_logits(labels = rho0, logits = rho0/tf.reduce_mean(h2,0)))
loss2 = tf.reduce_mean(tf.square(tf.nn.relu(tf.matmul(h2, tf.transpose(W2_en))+b2_de) - h1))+ beta * kl_div_loss2

loss3 = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#%%
train_step1_optimizer = tf.train.GradientDescentOptimizer(learning_rate= learning_rate1)
grads_and_vars1 = train_step1_optimizer.compute_gradients(loss1,var_list=[W1_en, b1_en, b1_de])
capped_grads_and_vars1 = [(tf.clip_by_value(gv[0],-1,1),gv[1]) for gv in grads_and_vars1]
train_step1=train_step1_optimizer.apply_gradients(grads_and_vars1)
#tf.Print(capped_grads_and_vars1[0])


train_step2_optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate2)
grads_and_vars2 = train_step2_optimizer.compute_gradients(loss2,var_list=[W2_en, b2_en, b2_de])
capped_grads_and_vars2 = [(tf.clip_by_value(gv[0],-1,1),gv[1]) for gv in grads_and_vars2]
train_step2=train_step2_optimizer.apply_gradients(grads_and_vars2)


train_step3 = tf.train.GradientDescentOptimizer(learning_rate3).minimize(loss3, var_list=[W3, b3])
#gradient descent: gradient of loss function with respect to weights
#take steps in opposite direction of gradient to minimize loss

#%%
# Import Tensorflow debug
sess = tf.InteractiveSession()
#from tensorflow.python import debug as tf_debug
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
sess.run(tf.global_variables_initializer())


# Train
for i in range(iteration):
    batch_xs, batch_ys= mnist.train.next_batch(batch_size)
    sess.run(train_step1, feed_dict={x: batch_xs})

for i in range(iteration):
    batch_xs, batch_ys= mnist.train.next_batch(batch_size)
    sess.run(train_step2, feed_dict={x: batch_xs})

for i in range(iteration):
    batch_xs, batch_ys= mnist.train.next_batch(batch_size)
    train_step3.run(feed_dict={x: batch_xs, y_: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))
#%%
# Visualize filters

# pixels = W1_en[:,1].eval() # convert tensor to array
# pixels = pixels.reshape((28, 28))
# imshow(pixels, cmap = 'winter')
# show()

# pixels1 = W1_en[:,2].eval() # convert tensor to array
# pixels1 = pixels1.reshape((28, 28))
# imshow(pixels1, cmap = 'winter')
# show()

# pixels2 = W1_en[:,3].eval() # convert tensor to array
# pixels2 = pixels2.reshape((28, 28))
# imshow(pixels2, cmap = 'winter')
# show()
#%%