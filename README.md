# Introduction to Deep Learning

### [View Presentation](http://donaldwhyte.github.io/intro-to-deeplearning/)

Presentation which gives a high-level overview of deep learning, using coding
examples to illustrate how to apply deep learning in practice.

Topics covered:

* machine learning definition
* growth of adoption
* supervised learning
    - classification
    - regression
    - feature extraction
    - model types and overfitting
* neural networks
    - perceptron
    - activation functions
    - feed-forward neural networks
    - backpropagation
    - vanishing gradient problem
* deep learning
    - training deep networks
    - types of deep networks
    - applying deep learning to high-energy physics problem

## Running Presentation

You can also run the presentation on a local web server. Clone this repository and run the presentation like so:

```
npm install
grunt serve
```

The presentation can now be accessed on `localhost:8080`. Note that this web application is configured to bind to hostname `0.0.0.0`, which means that once the Grunt server is running, it will be accessible from external hosts as well (using the current host's public IP address).

