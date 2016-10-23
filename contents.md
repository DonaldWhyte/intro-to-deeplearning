A bit about myself...

[NEXT]
<!-- .slide: data-background="images/bloomberg.jpg" class="stroke large" -->

![bloomberg](images/bloomberg_logo.svg)

Infrastructure Engineer

_note_
I currently work for Bloomberg as an infrastructure engineer. I help design
and build the low-level systems that keep financial data moving to the right
places.

My role is essentially a hybrid between a software engineer, architect and
data scientist. I dabble in a bit of everything basically!

[NEXT]
<!-- .slide: data-background="images/stbuk.jpg" class="stroke large" -->

Hackathons

Organiser / Mentor / Hacker

_note_
Have participated in, mentored at and organised 17 hackathons, across the world.

In countries such as: Egypt, UAE, Italy, Germany, the US and, of course, the UK.

[NEXT]
![bt](images/bt.svg)
![ibm](images/ibm.svg)
![helper.io](images/helper.svg)

Applied machine learning in:

* network security
* enterprise software management
* employment

_note_
I've applied machine learning to

increase global network security

assist organisations maintain their large, enterprise software estates

and more recently, I've consulted for recruitment startup helper.io


[NEXT SECTION]
## Machine Learning

#### Very Quick Overview

[NEXT]
**Machine learning** is an approach to achieve AI

Machines learn behaviour with little human intervention <!-- .element: class="fragment" data-fragment-index="1" -->

Programs that can adapt when exposed to new data <!-- .element: class="fragment" data-fragment-index="2" -->

Based on pattern recognition <!-- .element: class="fragment" data-fragment-index="3" -->

[NEXT]
![education](images/application1.jpg)
![security](images/application2.jpg)
![robotics](images/application3.jpg)
![finance](images/application4.jpg)
![speech recognition](images/application5.jpg)
![advertising](images/application6.jpg)

_note_
Machine learning has been around for a long time, but industry adoption and
academic research has grown rapidly the past five to ten years.

It's now being used in:

education
security
robotics
finance
speech recognition
image recognition
advertising

[NEXT]
<!-- .slide: data-background="images/ml-landscape-dec15.jpg" -->

_note_
TODO

[NEXT]
### Learning Types

* supervised learning
* unsupervised learning
* semi-supervised learning
* reinforcement learning

Supervised learning will be covered here <!-- .element: class="fragment" data-fragment-index="1" -->

[NEXT]
### Supervised Learning

Use labelled historical data to predict future outcomes

[NEXT]
Given some input data, predict the correct output

![shapes](images/shapes.svg)

What **features** of the input tell us about the output?

[NEXT]
### Feature Space

* A feature is some property that describes raw input data</li>
* Features represented as a vector in **feature space**
* **Abstract** complexity of raw input for easier processing

<center>
  <div id="shape-plot"></div>
</center>

_note_
In this case, we have 2 features, so inputs are 2D vector that lie in
a 2D feature space.

[NEXT]
### Classification

<div class="left-col">
  <ul>
    <li>Training data is used to produce a model</li>
    <li> *f(x&#x0304;)* = *mx&#x0304;* + *c*</li>
    <li>Model divides feature space into segments</li>
    <li>Each segment corresponds to one <strong>output class</strong></li>
  </ul>
</div>

<div class="right-col">
  <center>
    <div id="shape-plot-discriminant"></div>
  </center>
</div>

<div class="clear-col"></div>

<p>
  Use trained model to classify new, unseen inputs
</p>

[NEXT]
### Model Complexity and Overfitting

[NEXT]
<div class="left-col">
  <center>
    <div id="shape-plot-complex"></div>
  </center>
</div>
<div class="right-col fragment" data-fragment="1">
  <center>
    <div id="shape-plot-overfitting"></div>
  </center>
</div>

_note_
Now in reality, your data might not be linearly separable, so we might have
to use a more complex model to correctly discriminate between the different
output classes.

Of course, we need to be careful our models don't overfit our input training
data, otherwise it will fail to correctly classify new, unseen data points.

We can see on the diagram on the right, there are many unseen square instances
that have been incorrectly classified as a triangle. The two triangles near
them might just be outliers, but because the model was trained on a small
training dataset, the feature space looked like it had a different structure.

[NEXT SECTION]

## Neural Networks

[NEXT]
<!-- .slide: data-background-video="videos/neuron.mp4" data-background-video-loop="loop" data-background-video-muted="muted" -->

_note_
TODO

[NEXT]
<!-- .slide: data-background-video="videos/neural_networks.mp4" data-background-video-loop="loop" data-background-video-muted -->

_note_
TODO

[NEXT]
### The Mighty Perceptron

* Algorithm for supervised learning
* Modelled after a neuron in the human brain
* Linear classifier
    - splits feature space using a straight line

[NEXT]
<!-- .slide: data-background-color="white" -->
![perceptron](images/perceptron.svg)

_note_
TODO

[NEXT]
### Perceptron Definition

For `n` features, the perceptron is defined as:

* `n`-dimensional weight vector `w` 
* bias scalar `b`
* activation function `f(x)`

[NEXT]
Input: $$ x = \left(x_0, x_1, \cdots, w_n\right) $$

Weights: $$ w = \left(w_0, w_1, \cdots, w_n\right) $$

Bias: $b$

Weighted Sum:

$$ s = \left(\sum_{i=0}^{n} {w_ix_i}\right) + b $$

[NEXT]
### Activation Function

Simulates the 'firing' of a physical neuron

1 = neuron fired

0 = neuron did not fire

<div class="fragment fade-in" data-fragment-index="1">
<div class="fragment fade-out" data-fragment-index="2">
$$
  f(x) = \begin{cases}1 & \text{if }s > 0\\\\0
  & \text{otherwise}\end{cases}
$$
</div>
</div>

<div class="fragment fade-in" data-fragment-index="2">
$$
  f(x) = \begin{cases}1 & \text{if }w \cdot x + b > 0\\\\0
  & \text{otherwise}\end{cases}
$$
</div>

[NEXT]
### Step Function

* Produces binary output from real-valued sum
* Used for for binary classification
  - e.g. Triangle (0) or Square(1)

PLACEHOLDER<!-- .element id="step-activation-function-chart" -->

[NEXT]
### Sigmoid Function

* Can make perceptron produce continuous output
* Useful for regression (predicting continuous values)
  - e.g. temperature

PLACEHOLDER<!-- .element id="all-activation-functions-chart" -->

_note_
We'll find having a continuous activation function very useful for when we
combine many perceptrons together. 

[NEXT]
How do we find `w` and `b`?

[NEXT]
### Perceptron Learning Algorithm

Takes training dataset (known input/output pairs)

Algorithm which learns correct weights and bias

Guaranteed to create line that splits classes

(if data is linearly separable)<!-- .element class="small" -->

_note_
Details of the algorithm are not covered here for brevity

[NEXT]
<!-- .slide: data-background-color="white" data-transition="none" -->
![perceptron_learning](images/perceptron_learning1.png)

[NEXT]
<!-- .slide: data-background-color="white" data-transition="none" -->
![perceptron_learning](images/perceptron_learning2.png)

[NEXT]
<!-- .slide: data-background-color="white" data-transition="none" -->
![perceptron_learning](images/perceptron_learning3.png)

[NEXT]
<!-- .slide: data-background-color="white" data-transition="none" -->
![perceptron_learning](images/perceptron_learning4.png)

[NEXT]
### Problem

Most data is not linearly separable

Need a *network* of neurons to discriminate non-linear data

[NEXT]
### Feed-Forward Neural Networks

TODO

[NEXT]
### Training FFNNs

TODO

[NEXT]
### Backpropogation

TODO

[NEXT]
### Demo

TODO?

[NEXT]
### Problems

TODO: feature engineering + data cleaning (latter is an aside)

[NEXT SECTION]
## Deep Learning
#### What is It?

A machine learning technique

Based on neural networks

**Learns** the input features for you

[NEXT]
<!-- .slide: data-background="images/nvidia_deep_learning.png" -->

[NEXT]
### Issues with Deep Learning
##### (until 2006)

* Computationally expensive to train
* Models have a tendency to overfit training data
* Lack of *convergence* in weights
* Meant deep networks were useless in the real world
    - no better than random initialisation of weights

_note_
The earliest layers in a deep network simply weren't learning useful
representations of the data. In many cases they weren't learning anything at
all.

Instead they were staying close to their random weight initialisation because
of the nature of backpropagation. This meant the deep networks were completely
useless when used on real world data. They just didn't generalise at all.

[NEXT]
### Resurgence of Neural Networks

2006 and beyond

<table>
    <tr>
        <td>![Geoff Hinton](images/geoff_hinton.jpg)</td>
        <td>![Yann Lecun](images/yann_lecun.jpg)</td>
        <td>![Yoshua Bengio](images/yoshua_bengio.jpg)</td>
    </tr>
    <tr>
        <td>Geoff Hinton</td>
        <td>Yann Lecun</td>
        <td>Yoshua Bengio</td>
    </tr>
    <tr>
        <td>Restricted Boltzmann<br />Machine</td>
        <td>Sparse Representations</td>
        <td>Stacked Autoencoders</td>
    </tr>
</table>

_note_
In 2006 three separate groups developed ways of overcoming the difficulties
that many in the machine learning world encountered while trying to train
deep neural networks. The leaders of these three groups are the fathers of
the age of deep learning. 

Before their work, the earliest layers in a deep network simply weren’t
learning useful representations of the data. In many cases they weren’t
learning anything at all.

Using different techniques, each of these three groups was able to get these
early layers to learn useful representations, which led to much more powerful
neural networks.

[NEXT]
### Resurgence of Neural Networks

Google's usage that popularised it (TODO: get usage)

[NEXT]
### Modern Uses of Deep Learning

Widely used in image recognition

TODO: examples (driver-less cards and two others)

[NEXT]
### Other Factors

* cheaper computing power and data storage
* more data than ever &dash; everyone is online
* GPGPUs

[NEXT]
#### Projected Deep Learning Software Revenue

Placeholder<!-- .element id="deep-learning-growth-chart" -->  

[Source: Tractica](http://bit.ly/1Sufnue) <!-- .element class="small" -->

[NEXT]
> Annual revenue for deep learning will surpass **$10 billion** by 2014

_note_
The market intelligence firm forecasts that annual software revenue for
enterprise applications of deep learning will increase from $109 million
in 2015 to $10.4 billion in 2024.

I suspect this large growth is because of deep learning's application in
driver-less cars, which is going to become a *huge* industry.

[NEXT]
TODO


[NEXT SECTION]
## Deep Learning

#### Technical Overview

[NEXT]
TODO


[NEXT SECTION]
## Tensorflow
#### Deep Learning in Practice

[NEXT]
TODO


[NEXT SECTION]
## Summary

[NEXT]
TODO
