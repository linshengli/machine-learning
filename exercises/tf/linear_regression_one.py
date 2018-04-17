import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

size = 80
e_pochs = 5000

data = pd.read_csv("/home/tbxsx/Code/learnMachineLearning/exercises/tf/data/ex1data1.txt", header=None)


trainData = data.ix[data.index < size, :]
testData = data.ix[data.index > size, :]

X = tf.placeholder(dtype=tf.float32, shape=[size, ], name="X")
Y = tf.placeholder(dtype=tf.float32, shape=[size, ], name="Y")

W = tf.Variable(tf.random_normal(shape=[1]), dtype=tf.float32, name="W")
b = tf.Variable(tf.random_normal(shape=[1]), dtype=tf.float32, name="b")

Y_pred = tf.add(tf.multiply(W, X), b, name="Y_pred")
loss = tf.reduce_sum(tf.square(Y_pred - Y), name="loss") / size

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)
losses = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("/home/tbxsx/Code/learnMachineLearning/exercises/graph1", graph=sess.graph)
    for j in range(e_pochs):
        _x, _y = trainData.ix[:, 0], trainData.ix[:, 1]
        _, l, W1, b1 = sess.run([optimizer, loss, W, b], feed_dict={X: _x, Y: _y})
        losses.append(l)
        if j % 20 == 0:
            print("Epoch {0}: {1}:{2},{3}".format(j, l, W1, b1))
    W, b = sess.run([W, b])
    writer.close()

# plot the results
X, Y = trainData.ix[:, 0], trainData.ix[:, 1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * W + b, 'r', label='Predicted data')
plt.figure()
plt.plot(losses)
plt.show()
