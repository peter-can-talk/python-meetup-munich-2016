"""
FizzBuzz in TensorFlow.

Based on (https://github.com/joelgrus/fizz-buzz-tensorflow).
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from collections import namedtuple


Layer = namedtuple('Layer', 'weights, bias')
Data = namedtuple('Data', 'examples, labels')


class FizzBuzzer(object):
    """Fizzes, then buzzes."""

    CLASSES = 4

    def __init__(self, digits, hidden_units=100, batch_size=128):
        """
        Initialize the FizzBuzzer.

        Arguments:
            digits (int): Maximum number of digits the numbers have.
            hidden_units(int): The number of units to use for the hidden layer.
            batch_size(int): The batch size for SGD.
        """
        self.digits = digits
        self.hidden_units = hidden_units
        self.batch_size = batch_size

        # Handles
        self.optimize = None
        self.predictions = None

        self.loss_tensor = None
        self.loss = None

        self.accuracy_tensor = None
        self.accuracy = None

        self.training_data = self.generate_training_data(digits)
        self.graph = self.setup_graph()
        self.session = None

    def predict(self, number):
        """
        Determine if a number fizzes or buzzes, or maybe both.

        Arguments:
            number (int): The number to compute the fizzy-buzziness for.

        Returns:
            'Fizz' if the number is divisble by 3, 'Buzz' if it is divisble by
            5, 'FizzBuzz' if it is divisible by both 3 and 5 (15) and
            otherwise the number itself. Hopefully.
        """
        example, label = self.preprocess(number)
        predictions = self.session.run(self.predictions, feed_dict={
            self.examples: [example],
            self.labels: [label]
        })

        return self.decode(number, predictions)

    def train(self):
        """Teach the FizzBuzzer how to fizz, buzz or both."""
        batch = self.sample_batch(self.training_data, self.batch_size)
        fetches = [self.optimize, self.loss_tensor, self.accuracy_tensor]
        result = self.session.run(fetches, feed_dict={
            self.examples: batch.examples,
            self.labels: batch.labels
        })
        self.loss, self.accuracy = result[1:]

    def setup_graph(self):
        """Initialize a TensorFlow graph for fizzbuzzing."""
        graph = tf.Graph()
        with graph.as_default():
            # The input
            self.examples = tf.placeholder(tf.float32, [None, self.digits])
            self.labels = tf.placeholder(tf.float32, [None, self.CLASSES])

            # Create layers
            hidden_layer = self.create_layer(self.digits, self.hidden_units)
            output_layer = self.create_layer(self.hidden_units, self.CLASSES)

            # Propagate the data
            hidden = self.propagate(self.examples, hidden_layer, tf.nn.relu)
            scores = self.propagate(hidden, output_layer)

            # Predictions and loss
            self.predictions = tf.nn.softmax(scores)
            self.loss_tensor = self.cross_entropy(self.labels, self.predictions)

            # Gradient Descent
            gdo = tf.train.GradientDescentOptimizer(0.5)
            self.optimize = gdo.minimize(self.loss_tensor)

            self.accuracy_tensor = self.compute_accuracy(
                self.predictions,
                self.labels
            )

        return graph

    def propagate(self, data, layer, activation=None):
        """
        Propgate data through a neural network layer.

        Arguments:
            data (tf.Tensor): The tensor to propgate.
            layer (Layer): The layer to propagate through.
            activation (function): The activation function to use, if any.

        Returns:
            The result of propagating the data through the layer and applying
            the given activation function.
        """
        scores = tf.matmul(data, layer.weights) + layer.bias
        return activation(scores) if activation else scores

    def generate_training_data(self, digits):
        """
        Generate examples and labels for training.

        Arguments:
            digits (int): The number of digits to use for the data.

        Returns:
            All possible examples and associated labels for this many digits,
            as a Data object.
        """
        possible_values = 2**digits - 101
        examples = np.ndarray([possible_values, self.digits], np.float32)
        labels = np.ndarray([possible_values, FizzBuzzer.CLASSES], np.float32)
        for index in range(possible_values):
            examples[index], labels[index] = self.preprocess(index + 101)

        return Data(examples, labels)

    def preprocess(self, number):
        """
        Preprocess a given number.

        Arguments:
            number (int): The number to preprocess.

        Returns:
            The binary encoded example vector and one-hot-encoded label vector.
        """
        example = self.binary_encode(number, self.digits)
        label = self.fizz_buzz_encode(number)

        return example, label

    def binary_encode(self, number, digits):
        """
        Binary encodes a number.

        Note that the bits are returned in reverse order. But since we can pick
        our own representation, this doesn't matter.

        Arguments:
            number (int): The number to encode.
            digits (int): The number of digits to evaluate.

        Returns:
            An array containing the bits of the number.
        """
        assert len(bin(number)[2:]) <= digits
        return np.array([number >> d & 1 for d in range(digits)])

    def __repr__(self):
        """Return a string representation of the FizzBuzzer's current state."""
        return 'accuracy: {0}, loss: {1}'.format(self.accuracy, self.loss)

    def __enter__(self):
        """Open the FizzBuzzer's Session."""
        self.session = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.session.run(tf.initialize_all_variables())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the FizzBuzzer's Session."""
        self.session.close()
        if exc_type is not None:
            raise

    @staticmethod
    def decode(number, prediction):
        """
        Decode a prediction vector.

        Arguments:
            number (int): The number to decode.
            prediction (tf.Tensor): The prediction vector for the number.
        """
        index = np.argmax(prediction)
        return [str(number), 'Fizz', 'Buzz', 'FizzBuzz'][index]

    @staticmethod
    def sample_batch(data, size):
        """
        Sample a new random batch of the given data, with the given size.

        Arguments:
            data (Data): The data to sample from.
            size (int): How many data points to sample
        """
        indices = np.random.permutation(len(data.labels))[:size]
        return Data(data.examples[indices], data.labels[indices])

    @staticmethod
    def compute_accuracy(p, q):
        """
        Compute the accuracy between two probability distributions.

        Arguments:
            p (tf.Tensor): The first probability distribution.
            q (tf.Tensor): The second probability distribution.
        """
        equal = tf.equal(tf.argmax(p, dimension=1), tf.argmax(q, dimension=1))
        return tf.reduce_mean(tf.cast(equal, tf.float32))

    @staticmethod
    def cross_entropy(p, q):
        """
        Compute cross entropy between two probability distributions.

        Arguments:
            p (tf.Tensor): The first probability distribution.
            q (tf.Tensor): The second probability distribution.
        """
        batch = -tf.reduce_sum(p * tf.log(q), reduction_indices=1)
        return tf.reduce_mean(batch)

    @staticmethod
    def create_layer(input_units, output_units):
        """
        Create a new neural network layer.

        Arguments:
            input_units (int): The number of units going into the layer.
            output_units (int): The number of units going out of the layer.
        """
        weights = FizzBuzzer.create_weights([input_units, output_units])
        bias = FizzBuzzer.create_weights([output_units])

        return Layer(weights, bias)

    @staticmethod
    def create_weights(shape):
        """
        Create a weight tensor with the given shape.

        Samples from a random uniform distribution.

        Arguments:
            shape (int): The shape of the weights to create.
        """
        return tf.Variable(tf.random_normal(shape, stddev=0.01))

    @staticmethod
    def fizz_buzz_encode(number):
        """
        One-hot-encode a number.

        Arguments:
            number (int): The number to encode.

        Returns:
            A one-hot-encoded vector representation of the number.
        """
        if number % 15 == 0:
            return np.array([0, 0, 0, 1])
        elif number % 5 == 0:
            return np.array([0, 0, 1, 0])
        elif number % 3 == 0:
            return np.array([0, 1, 0, 0])
        else:
            return np.array([1, 0, 0, 0])


def main():
    number_of_epochs = 5000
    checkpoint = number_of_epochs / 10
    with FizzBuzzer(10) as fizz_buzz:
        for epoch in range(1, number_of_epochs + 1):
            fizz_buzz.train()
            if epoch % checkpoint == 0:
                print('Iteration {0}: {1}'.format(epoch, fizz_buzz))
        while True:
            number = input('Enter a number: ')
            print(fizz_buzz.predict(int(number)))

if __name__ == '__main__':
    main()
