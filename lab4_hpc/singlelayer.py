#!/usr/bin/env python
import tensorflow as tf
import read_inputs
import numpy as N
import matplotlib.pyplot as plt


#read data from file
data_input = read_inputs.load_data_mnist('MNIST_data/mnist.pkl.gz')
#FYI data = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
data = data_input[0]
print ( N.shape(data[0][0])[0] )
print ( N.shape(data[0][1])[0] )

#data layout changes since output should an array of 10 with probabilities
real_output = N.zeros( (N.shape(data[0][1])[0] , 10), dtype=N.float )
for i in range ( N.shape(data[0][1])[0] ):
  real_output[i][data[0][1][i]] = 1.0  

#data layout changes since output should an array of 10 with probabilities
real_check = N.zeros( (N.shape(data[2][1])[0] , 10), dtype=N.float )
for i in range ( N.shape(data[2][1])[0] ):
  real_check[i][data[2][1][i]] = 1.0



#set up the computation. Definition of the variables.
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step1 = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step2 = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
train_step3 = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
train_step4 = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
train_step5 = tf.train.RMSPropOptimizer(0.01).minimize(cross_entropy)


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#TRAINING PHASE
print("TRAINING 1")

cross_list1 = []
for i in range(500):
  batch_xs = data[0][0][100*i:100*i+100]
  batch_ys = real_output[100*i:100*i+100]
  sess.run(train_step1, feed_dict={x: batch_xs, y_: batch_ys})
  curr_W, curr_b, curr_cross = sess.run([W, b, cross_entropy], {x: batch_xs, y_: batch_ys})
  print("W: %s b: %s cross: %s"%(curr_W, curr_b, curr_cross))
  cross_list1.append(curr_cross)

#CHECKING THE ERROR
print("ERROR CHECK 1")

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: data[2][0], y_: real_check}))

print('cross_list1: {}'.format(cross_list1))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#TRAINING PHASE
print("TRAINING 2")

cross_list2 = []
for i in range(500):
  batch_xs = data[0][0][100*i:100*i+100]
  batch_ys = real_output[100*i:100*i+100]
  sess.run(train_step2, feed_dict={x: batch_xs, y_: batch_ys})
  curr_W, curr_b, curr_cross = sess.run([W, b, cross_entropy], {x: batch_xs, y_: batch_ys})
  print("W: %s b: %s cross: %s"%(curr_W, curr_b, curr_cross))
  cross_list2.append(curr_cross)

#CHECKING THE ERROR
print("ERROR CHECK 2")

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: data[2][0], y_: real_check}))

print('cross_list2: {}'.format(cross_list2))


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#TRAINING PHASE
print("TRAINING 3")

cross_list3 = []
for i in range(500):
  batch_xs = data[0][0][100*i:100*i+100]
  batch_ys = real_output[100*i:100*i+100]
  sess.run(train_step3, feed_dict={x: batch_xs, y_: batch_ys})
  curr_W, curr_b, curr_cross = sess.run([W, b, cross_entropy], {x: batch_xs, y_: batch_ys})
  print("W: %s b: %s cross: %s"%(curr_W, curr_b, curr_cross))
  cross_list3.append(curr_cross)

#CHECKING THE ERROR
print("ERROR CHECK 3")

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: data[2][0], y_: real_check}))

print('cross_list3: {}'.format(cross_list3))


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#TRAINING PHASE
print("TRAINING 4")

cross_list4 = []
for i in range(500):
  batch_xs = data[0][0][100*i:100*i+100]
  batch_ys = real_output[100*i:100*i+100]
  sess.run(train_step4, feed_dict={x: batch_xs, y_: batch_ys})
  curr_W, curr_b, curr_cross = sess.run([W, b, cross_entropy], {x: batch_xs, y_: batch_ys})
  print("W: %s b: %s cross: %s"%(curr_W, curr_b, curr_cross))
  cross_list4.append(curr_cross)

#CHECKING THE ERROR
print("ERROR CHECK 4")

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: data[2][0], y_: real_check}))

print('cross_list4: {}'.format(cross_list4))


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#TRAINING PHASE
print("TRAINING 5")

cross_list5 = []
for i in range(500):
  batch_xs = data[0][0][100*i:100*i+100]
  batch_ys = real_output[100*i:100*i+100]
  sess.run(train_step5, feed_dict={x: batch_xs, y_: batch_ys})
  curr_W, curr_b, curr_cross = sess.run([W, b, cross_entropy], {x: batch_xs, y_: batch_ys})
  print("W: %s b: %s cross: %s"%(curr_W, curr_b, curr_cross))
  cross_list5.append(curr_cross)

#CHECKING THE ERROR
print("ERROR CHECK 5")

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: data[2][0], y_: real_check}))

print('cross_list5: {}'.format(cross_list5))

# Plots
#plt.plot(loss_list, iteration_list, 'o-')
plt.plot(cross_list1, 'b-', label='Gradient Descent 0.5')
plt.plot(cross_list2, 'r-', label='Gradient Descent 0.1')
plt.plot(cross_list3, 'y-', label='Gradient Descent 0.01')
plt.plot(cross_list4, 'g-', label='Adam Optimizer 0.01')
plt.plot(cross_list5, 'k-', label='RMSProp Optimizer 0.01')
plt.xlabel('Iteration Number')
plt.ylabel('Cross Entropy Function Value')
#plt.title(networks[i] + ', SIS(μ=%.1f, P0=%.1f)' % (mu, p0))
plt.title('Exercise 2: MNIST Single Layer')
plt.legend(loc='upper right', shadow=False, fontsize='small')

plt.savefig('Exercise2.png')
plt.close()

