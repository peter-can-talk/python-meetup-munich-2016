# Study notes

## Induction vs Inference

*Inference* is the act of drawing (deriving) a local conclusion from some given premisses (truths) known or *assumed* to be true.

*Deductive Inference* (top-down logic) is the process of inferring specific knowledge given more general premisses. Because the premisses are (assumed to be) true and all inference must follow the rules of logic, the conclusion is a certain truth too.

*Induction Inference* (bottom-up logic) is the process of inferring more general knowledge given specific knowledge (examples). Because the specific can never really stand for the whole, all conclusions reached via inductive inference are only probable (given the evidence). The premisses are thus viewed to give *strong support* for the conslusion, but cannot make it certain.

" In deductive reasoning, a conclusion is reached reductively by applying general rules that hold over the entirety of a closed domain of discourse, narrowing the range under consideration until only the conclusion(s) is left".

"In inductive reasoning, the conclusion is reached by generalizing or extrapolating from specific cases to general rule"

## Biological Motivation

Brain has 86M neurons. Neurons have input wires -- *dendrites* --, nuclei and output wires -- *axons*. At the end of axons, there are *axon terminals*, which in turn connect via synapses to the dendrites of other neurons. The idea is that the output signals $x$ of one neuron, traveling along its axons, interact multiplicatively with the dendrites of other neurons (weighted edges). The degree of interaction is modeled by some weight $w$, such that the input to the nucleus of a neuron is $wx$. This weight (*synaptic strength*) is learned and control the influence one neuron has on another neuron. Positive weights model excitory connections, negative weighs model inhibiting connections. In the mathematical model, the weighted inputs are summed. If the final sum is greater than some threshold, the neuron will output its own signal on its axon, leading to other neurons. We model the firing rate or firing threshold via the activation function of the neuron (e.g. sigmoid or tanh).

This is *very* simplified and any neuroscientist will hate you for saying that neural networks are exactly like real neural activity in the brain.

## Convolutional NNs

http://colah.github.io/posts/2014-07-Conv-Nets-Modular/

The basic idea of a convolutional neural network is to share weights and parameters.

You can stack convolutional layers on top of each other. And you can have *pooling* layers in between, which perform a kind of filter or selection function on the data.

Convolutional neural network layers *learn* to detect things, such as edges or color contrasts. Patterns that match. Show picture from colah.

What the value of the weights in a neural network generally are, are measures of how happy the network is to see a certain value from some input unit. Positive weights encourage the input unit to take on (large) values (such as bright pixel intensities). Negative values inhibit or punish a unit from taking on values. It would rather that input (e.g. pixel) would be zero. This contributes to the result of the neuron "firing" or not, if we assume a neuron "fires" (like in the MLP) case. Also, the bias sets the threshold for when the neuron fires.

Stride, variants of padding (hyperparameters)

So you've just spent a lot of time understanding how ConvNets work for one dimensional data. Then someone comes along and gives you two-dimensional data (matrices) and says, well, "figure that one out". Then your response should be: "screw you and your stupid second dimension, I will flatten your matrix into a vector".

Non-linearity after the dot product.

Make the explicit assumption that the inputs are images, allowing for more specific solutions.

Reasons why convnets are better for images:
* Translation invariance via weight sharing
* Scale better, since images will have many features (pixels).
* Simplest thing we could do is flatten the image and feed it into a fully connected neural network.
* However, a $200 \times 200$ pixel image already means 40,000 pixels. Add to that three RGB layers, so that actually gives $200 \times 200 \times 3 = 120,000$ features. In a typical neural network we'd often have something like 100 units per hidden layer, so that would give us 12,000,000 parameters for one layer. And then we'd have many layers ... doesn't scale.
* Many parameters also lead to overfitting, since they will specialize more to their data.

pooling = downsampling

"the network will learn filters that activate when they see some type of visual feature such as an edge of some orientation or a blotch of some color on the first layer"

Generally the early layers will be looking for edges or blobs of color or color contrasts.

Three hyperparameters:

* Depth (how many filters)
* Stride: how many pixels to skip. Usually 1 or 2.
* Padding: Add zeros to the filter to control the width and height of the output.

These three parameters control the spatial arrangement of the output volume.

The filters should not alter the spatial dimensions of the volume (width and height, i.e. alter only the depth).

To preserve the size, the padding should be $P = \frac{F - 1}{2}$. The filter size should and must always be an odd number, it just doesn't work otherwise to preserve the size of the input volume. The reason why is that when we have an odd filter size, we always have some center pixel that we can put on top of every input pixel, ensuring that we will receive exactly one output for each pixel (when the center pixel is on top of it). So we know the filter must have an odd size. Then the padding must accommodate the condition that the center pixel is on each input pixel once. When the center pixel is at the very left or very right, exactly the half to the right of the center (median) will overlap. This is of course exactly $(F - 1)/2$ ($F$ minus the center pixel gives the left and right halves, divided by two the size of one such half).

So we know $F$ is odd and $2P$ is obviously even. We also know that to make the patch fit all input pixels, the output size $(W - F + 2P)/S + 1$ must be an integer. Since $-F + 2P$ will be odd (odd + even = odd), we must make $W$ and $S$ work together to produce an integer in the fraction. Meaning, when $W$ is odd the numerator will be even, so the stride must be even. When $W$ is even the numerator will be odd and the stride must be odd.

All neurons (duplicates of the "kernel neuron") of one layer use the same weights. During backprop, each duplicate (each position of the patch on the image as it is sliding across) will compute its own gradient, but they will be added up for each layer, and only one update will be made to the weights of that layer (the weights in the single feature map of that layer in the kernel).

The point of sharing parameters across an image is translational invariance. If detecting a horizontal edge is important in one part of an image, it is also important in other areas of the image. Because we want to detect that edge (or kind of edge) wherever it is. We therefore need not relearn to detect a horizontal edge at every patch location.

This may often not make sense, when the object of interest is actually not spread across the image. For example a face will usually be in the center. So we will relax the parameter sharing scheme and use a locally (not fully) connected layer.

Pooling layers reduce the spatial size of data volumes in a network. More parameters = more overfitting, so this also reduces risk of that.

Pooling operations do not modify the depth of the model, but only its width and height. Also helps translational invariance. Two common forms of pooling are:

1. $F = 2, S = 2$, meaning $2 \times 2$ patches with no overlap. This selects one pixel out of every four, reducing the spatial extent by 75%.
2. $F = 3, S = 2$, meaning $3 \times 3$ patches with overlap (same pixel might be pixel twice)

Other operations we can do in a pooling layer except for max pooling is average pooling (blurrs the image) or L2-norm pooling (root of the squares of the values).

AlexNet had 60M parameters. 10% less error than the runner up.

If we ensure that the filters are used with the right padding, they should not alter the spatial dimensions. In effect, only the pooling operations alone are responsible for altering the width and height. Convolutions (especially 1x1 convolutions) would only alter the depth.

ConvNet architectures:

`INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC`

We can visualize ConvNets with t-SNE using CNN-Codes (the activations just before the classifier, i.e. *before* the output layer.)

## Backpropagation

http://colah.github.io/posts/2015-08-Backprop/

Forward mode differentiation = computing the derivative of each output w.r.t. to one input.
Reverse mode differentiation = computing the derivative of one output w.r.t to each input.

"When we train neural networks, we don't find the global minima. Actually, we also don't even find a local minima: recent research suggests we get stuck on saddle points (eg. see some Ian Goodfellow and collaborators recent papers).

But, surprisingly, things still work really well! What's going on? Well, we still find good enough points that neural networks work well, even if they aren't minima."

## RNNs

http://colah.github.io/posts/2015-08-Understanding-LSTMs/

"As I am saying this sentence, my words only make sense to you in the context of the words I said before. So, I just used the word 'before' and if I just threw the word 'before' at you, without having any other words, it would not make sense to you. So if I meet you walking down the street and before I even say hi, I just say 'before', then you won't know what I'm talking about. So, clearly, to understand sequences such as those in natural language, our learning algorithm needs some sort of memory." ConvNets for example, can't do this.

RNNs are NNs with loops in their units.

Backprop through time = unrolling the unit in time and doing backprop.

We're again sharing weights, now over time.

We theoretically need to backprop through the time the entire sequence, but in practice we just go as far back as we can afford.

gradient clipping is also a way of hacking vanishing/exploding gradient. We compute the norm of the weights and when they get too large or small we just stop.

show forgetting the past as wobbly past.
