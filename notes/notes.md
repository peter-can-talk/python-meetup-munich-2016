# Talk structure

The talk will generally be structured into two parts:

1. Theory and general information on Machine and Deep Learning
2. TensorFlow and practical walkthrough

## Meta Elements ~ 10 mins

The talk's name will be *"Introduction to Machine Learning with TensorFlow"*.

-- First: show video of pianist (motivation)

1. Table of Contents (Catents)
  * Theory Cat
  * Practice Cat

Only talk to the parts, don't give an explicit table of contents (describe what you will be talking about in words).

2. Background
  * Describe myself (TUM, Google)
  * Describe why I'm giving this talk (seminar)
  * "Last time I had 15 people forced to listen to me so it's nice to see that I managed to force more than a hundred in this time. Although apparently some of you actually came voluntarily."

## Theory

The theory: what is machine learning and what can you do with it.

### Defining stuff ~20 mins

Define Machine Learning in a sentence, separate it from data science / data mining / AI.

Deep Learning Book (learn task given ...)

"Machine learning is not magic; it can’t get something from nothing. What it does is get more from less. Programming, like all engineering, is a lot of work: we have to build everything from scratch. Learning is more like farming, which lets nature do most of the work. Farm- ers combine seeds with nutrients to grow crops. Learners combine knowledge with data to grow programs."

--

Supervised vs unsupervised learning

--

What kinds of problems can you learn (classification, regression etc. + examples)

--

### Methods

Explain methods and steps of machine learning by example of linear regression.

--

Elements of a linear regression task:

Iris data set

* Trying to find an approximation $f$ of some target function $f^\star$
* Multiple features, where each feature is an axis in space
* Each feature should contribute some certain amount to determine the output of the function
* So we weight them. Also, it might be necessary to add a bias $b$.

--

Training, Validation, Test.

Cross validation.

http://stats.stackexchange.com/questions/19048/what-is-the-difference-between-test-set-and-validation-set

--

Softmax to turn values into probabilities
Error functions

--

Gradient Descent
Stochastic Gradient Descent

This here, is the machine learning.

--

Over/underfitting, regularization.

Deep learning book as source.

--

Non-linear functions: XOR. Add quadratic terms to regression function.

### Neural Networks ~ 10 mins

### Background & History

Who invented them, MLP

### The five levels of neural Networks

1. Biological Level. Explain neuroscientific motivation

2. The layer level: Perform some operation, apply some activation function.

3. The unit level. Weights, biases, individual function applications.

4. The function level. We are actually just composing functions.

5. The matrix level. The actual matrix multiplications.

Probabs show only a single layer to differentiate from deep learning.

### Backpropagation

Simple explanation (chain rule)

### Deep Learning ~ 20 mins

First slide: The why, the what and the ugly. I haven't thought of what the ugly is yet ...

### The what

Blunt definition: more than one layer.

Theoretically, we can approximate any function with a single layer. Exponentially many units (number of values^N). We could just use an infinitely sized hash table for our hidden layer and we'd have the best neural network in the world. Unfortunately, infinity does not tend scale very well, even with Hadoop, so don't try this out at home.

### The why

To answer why deep learning is good for certain tasks, let's see what problems learning algorithms can have. One very important problem is the curse of dimesionality. Values scale exponentially. We do not have enough data to train our model!

* Need to make assumptions ("priors") about our data to reduce the space of possible outputs (classes).
* The first step is to assume the data lies only in a subspace of the possible output space. So if we're trying to map images to 10-letter captions, we'd theoretically have an output space of $26^{10}$ classes (words), assuming only characters from the Latin alphabet. However, clearly, not all strings of 10 characters are valid words. Similarly, pixels in an image are not uniformly distributed.
* Furthermore, we can produce depenencies between regions in our dimensionality space.
* Local constancy prior is one example. Explain for k-nearest neighbor algorithm.
* "If we know a good answer for $x$, that answer will probably also be good for points in the neighborhood of $x$".
* This assumption works very well as long as we have enough training examples to fit each possible neihborhood.
* For very complex functions, this does not work.
* Deep nets make assumptions about data being hierarchical, being a composition of factors

Side note: No free lunch theorem!

### Why deep learning has become so succesful

* I have explained why deep learning is a good idea in theory, but why has it taken so long?
* Neural networks are not new
* Even fancy methods such as convolutional NNs are old (1990s, LeCun)
* Three reasons why deep learning has recently become so succesful:

1. Better hardware. Really emphasize this.
  * Faster CPUs and __especially__ GPUs. NVidia is making so much money. TPU
  * Bigger clusters
  * Enable learning of massive models on equally massive data sets
2. More data
  * Greater availability of data
  * 1990s they didn't have this Internet thing
  * with entire social networks full of selfies, status updates people think other people care. Just look at Google images. Billions of images.
  * Data scarcity still is a problem in some domains, like medicine.
3. Better methods:
  1. ReLu (explain)
  2. Dropout (explain)
  3. Softmax

### Kinds of Neural Networks

Quickly show different neural networks
Case studies w/ real world examples

### FeedForward Nets

Longer feedforward nets

### ConvNets

Convolutions in general
Max pooling

Case Study AlexNet

### RNNs

#### LSTMs

## Practice

TensorFlow talk ...

## FizzBuzz or MNIST
