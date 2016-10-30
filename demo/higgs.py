#!/usr/bin/env python3

from __future__ import print_function

import math
import tensorflow as tf
from dataset_loader import DatasetLoader

# Load training data
def loadDataset():
    inputs = []
    outputs = []
    with open('HIGGS.csv') as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        for row in reader:
            outputs.append(float(row[0]))
            inputs.append([float(x) for x in row[1:]])
    return inputs, outputs

def splitDatasetIntoTrainAndTest(inputs, outputs, numTrainingSamples):
    desiredClassA = numTrainingSamples / 2
    desiredClassB = numTrainingSamples / 2
    numClassA = 0
    numClassB = 0
    trainingInputs = []
    trainingOutputs = []
    testInputs = []
    testOutputs = []

    for i in range(len(inputs)):

    while len(trainingInputs) < numTrainingSamples:

        if numClassA <
        testInputs
    return trainingInputs, trainingOutputs, testInputs, testOutputs



# Parameters
training_epochs = 50
batch_size = 100
display_step = 1

# Network Parameters
hiddenLayerSizes = [ 500, 500, 500, 500 ]
numFeatures = 21
numClasses = 2

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
nodeKeepProbability = tf.placeholder(tf.float32)  # for dropout
learningRate = tf.placeholder(tf.float32)         # for decaying learning rate

# Stored weights and biases
weights = {
    'h1': tf.Variable(tf.random_normal(
        [n_input, hiddenLayerSizes[0]])),
    'h2': tf.Variable(tf.random_normal(
        [hiddenLayerSizes[0], hiddenLayerSizes[1]])),
    'h3': tf.Variable(tf.random_normal(
        [hiddenLayerSizes[1], hiddenLayerSizes[2]])),
    'h4': tf.Variable(tf.random_normal(
        [hiddenLayerSizes[2], hiddenLayerSizes[3]])),
    'out': tf.Variable(tf.random_normal(
        [hiddenLayerSizes[3], numClasses]]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([hiddenLayerSizes[0]])),
    'b2': tf.Variable(tf.random_normal([hiddenLayerSizes[1]])),
    'b3': tf.Variable(tf.random_normal([hiddenLayerSizes[2]])),
    'b4': tf.Variable(tf.random_normal([hiddenLayerSizes[3]])),
    'out': tf.Variable(tf.random_normal([numClasses]))
}

def createNetwork(x, weights, biases):
    # Four hidden layers with RELU activation. Apply dropout to each layer to
    layer_1 = tf.nn.relu(tf.matmul(x, weights['h1']) + biases['b1'])
    layer_1 = tf.nn.dropout(layer_1, nodeKeepProbability)
    layer_2 = tf.nn.relu(tf.matmul(layer_1, weights['h2']) + biases['b2'])
    layer_2 = tf.nn.dropout(layer_2, nodeKeepProbability)
    layer_3 = tf.nn.relu(tf.matmul(layer_2, weights['h3']) + biases['b3'])
    layer_3 = tf.nn.dropout(layer_3, nodeKeepProbability)
    layer_4 = tf.nn.relu(tf.matmul(layer_3, weights['h4']) + biases['b4'])
    layer_4 = tf.nn.dropout(layer_4, nodeKeepProbability)
    # Output layer with linear activation
    out_layer = tf.matmul(later_4, weights['out']) + biases['out']
    return out_layer


# Construct model
pred = creareModel(x, weights, biases)

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
