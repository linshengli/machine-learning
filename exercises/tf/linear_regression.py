import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

n_epochs = 50000
trainSize = 450
learning_rate = 0.0000001
input = pd.read_csv("/home/tbxsx/Code/learnMachineLearning/exercises/tf/boston.csv")

featureNum = input.shape[1]
sampleNum = input.shape[0]
m = tf.constant(input.shape[0], name="sampleNum")
feature = tf.constant(input.shape[1], name="featureNum")
trainX = tf.placeholder(shape=[trainSize, featureNum - 1], dtype=tf.float32)
trainY = tf.placeholder(shape=[trainSize, 1], dtype=tf.float32)

w = tf.Variable(tf.random_normal(shape=[featureNum - 1, 1], mean=0.0, stddev=1.0), name="weight", dtype=tf.float32)
b = tf.Variable(tf.random_normal(shape=[1], mean=0.0, stddev=1.0), name="bias", dtype=tf.float32)

Y_pred = tf.add(tf.matmul(trainX, w), b, name="Y_pred")
cost = tf.reduce_sum(tf.square(Y_pred - trainY)) / trainSize

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
losses = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter("/home/tbxsx/Code/learnMachineLearning/exercises/graph", graph=sess.graph)
    mytrainX = input.ix[input.index < trainSize, input.index < featureNum - 1]
    mytrainY = input.ix[input.index < trainSize, input.index == featureNum - 1]

    for epochs_i in range(n_epochs):
        _, c, W1, b1 = sess.run([optimizer, cost, w, b], feed_dict={trainX: mytrainX, trainY: mytrainY})
        if epochs_i % 100 == 0:
            print("Epoch {0} : {1} : {2}".format(epochs_i, c, b1[0]))
        losses.append(c)
    writer.close()

# plot

plt.plot(losses)
plt.show()
