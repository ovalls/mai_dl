#!/usr/bin/env python
import tensorflow as tf
import matplotlib.pyplot as plt


# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer1 = tf.train.GradientDescentOptimizer(0.01)
optimizer2 = tf.train.GradientDescentOptimizer(0.001)
optimizer3 = tf.train.AdamOptimizer(0.01)
optimizer4 = tf.train.RMSPropOptimizer(0.01)
train1 = optimizer1.minimize(loss)
train2 = optimizer2.minimize(loss)
train3 = optimizer3.minimize(loss)
train4 = optimizer4.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong

curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

loss_list1 = []
for i in range(1000):
    sess.run(train1, {x: x_train, y: y_train})
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
    loss_list1.append(curr_loss)

sess = tf.Session()
sess.run(init) # reset values to wrong

curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

loss_list2 = []
for i in range(1000):
    sess.run(train2, {x: x_train, y: y_train})
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
    loss_list2.append(curr_loss)

sess = tf.Session()
sess.run(init) # reset values to wrong

curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

loss_list3 = []
for i in range(1000):
    sess.run(train3, {x: x_train, y: y_train})
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
    loss_list3.append(curr_loss)


sess = tf.Session()
sess.run(init) # reset values to wrong

curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

loss_list4 = []
for i in range(1000):
    sess.run(train4, {x: x_train, y: y_train})
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
    loss_list4.append(curr_loss)

# Plots
#plt.plot(loss_list, iteration_list, 'o-')
plt.plot(loss_list1, 'b-', label='Gradient Descent 0.01')
plt.plot(loss_list2, 'r-', label='Gradient Descent 0.001')
plt.plot(loss_list3, 'g-', label='Adam Optimizer 0.01')
plt.plot(loss_list4, 'k-', label='RMSProp Optimizer 0.01')
plt.xlabel('Iteration Number')
plt.ylabel('Loss Function Value')
#plt.title(networks[i] + ', SIS(μ=%.1f, P0=%.1f)' % (mu, p0))
plt.title('Exercise 1: Optimization Problem')
plt.legend(loc='upper right', shadow=False, fontsize='small')

#plt.savefig(networks[i] + '_' + str(mu) + '.png')
plt.savefig('Exercise1.png')
plt.close()

