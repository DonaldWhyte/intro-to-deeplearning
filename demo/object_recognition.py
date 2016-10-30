#!/usr/bin/env python3

from __future__ import print_function

import math
import tensorflow as tf
from dataset_loader import DatasetLoader

# Training Data
IMAGE_DIMENSIONS = (100, 100)
datasetLoader = DatasetLoader(
    trainingSampleRate=0.5,
    imageDimensions=IMAGE_DIMENSIONS,
    maxLabels=10,
    preload=True)
testInputs, testOutputs = datasetLoader.testData()

# Parameters
training_epochs = 50
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 512 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_hidden_3 = 128 # 3rd layer number of features
n_input = IMAGE_DIMENSIONS[0] * IMAGE_DIMENSIONS[1] * 3 # input for each pixel
n_classes = datasetLoader.numLabels()

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
nodeKeepProbability = tf.placeholder(tf.float32)
learningRate = tf.placeholder(tf.float32)


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.nn.relu(tf.matmul(x, weights['h1']) + biases['b1'])
    layer_1 = tf.nn.dropout(layer_1, nodeKeepProbability)
    # Hidden layer with RELU activation
    layer_2 = tf.nn.relu(tf.matmul(layer_1, weights['h2']) + biases['b2'])
    layer_2 = tf.nn.dropout(layer_2, nodeKeepProbability)
    # Hidden layer with softmax activation
    layer_3 = tf.nn.softmax(tf.matmul(layer_2, weights['h3']) + biases['b3'])
    layer_3 = tf.nn.dropout(layer_3, nodeKeepProbability)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        averageCost = 0.
        batchIterator = datasetLoader.trainingIterator(batch_size)
        totalBatches = batchIterator.totalBatches()

        # Learning rate decay
        maxLearningRate = 0.003
        minLearningRate = 0.0001
        decaySpeed = 2000.0 # 0.003-0.0001-2000=>0.9826 done in 5000 iterations
        currentLearningRate = minLearningRate + \
            (maxLearningRate - minLearningRate) * math.exp(-epoch / decaySpeed)

        # Loop over all batches
        for batchInput, batchOutput in batchIterator:
            # Run optimization op (backprop) and cost op (to get loss value)
            _, currentCost = sess.run([optimizer, cost], feed_dict={
                x: batchInput,
                y: batchOutput,
                nodeKeepProbability: 0.9,
                learningRate: currentLearningRate
            })
            # Compute average loss
            averageCost += currentCost / totalBatches
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost={:.9f}".format(averageCost),
                  "learningRate={:.9f}".format(currentLearningRate))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({
        x: testInputs,
        y: testOutputs,
        nodeKeepProbability: 1.0  # no dropout during testing
    }))
