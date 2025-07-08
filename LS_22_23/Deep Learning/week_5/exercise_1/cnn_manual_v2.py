#!/usr/bin/env python3
import argparse
import os
from typing import List, Tuple
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

# Supr Dupr Extra Uzasny (sleep deprived) Tym:
# 3d76595a-e687-11e9-9ce9-00505601122b
# 594215cf-e687-11e9-9ce9-00505601122b

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--cnn", default="5-3-2,10-3-2", type=str, help="CNN architecture.")  # "5-3-2,10-3-2"
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--verify", default=True, action="store_true", help="Verify the implementation.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Convolution:
    def __init__(self, filters: int, kernel_size: int, stride: int, input_shape: List[int], verify: bool) -> None:
        # Create a convolutional layer with the given arguments
        # and given input shape (e.g., [28, 28, 1]).
        self._filters = filters
        self._kernel_size = kernel_size
        self._stride = stride
        self._verify = verify

        # Here the kernel and bias variables are created
        self._kernel = tf.Variable(tf.initializers.GlorotUniform(seed=42)(
            [kernel_size, kernel_size, input_shape[2], filters]))
        self._bias = tf.Variable(tf.initializers.Zeros()([filters]))
        #print(input_shape)

    def forward(self, inputs: tf.Tensor) -> tf.Tensor:
        # TODO: Compute the forward propagation through the convolution
        # with `tf.nn.relu` activation, and return the result.
        #
        # In order for the computation to be reasonably fast, you cannot
        # manually iterate through the individual pixels, batch examples,
        # input filters, or output filters. However, you can manually
        # iterate through the kernel size.
        (B, H, W, C) = inputs.shape
        #print("Input shape:", inputs.shape)
        k = self._kernel_size
        s = self._stride
        shape_2 = int(np.ceil((H - (k - 1)) / s))
        shape_3 = int(np.ceil((W - (k - 1)) / s))
        output = tf.zeros(shape=[B, shape_2, shape_3, self._filters])
        output += self._bias


        for m in range(self._kernel_size):
            for n in range(self._kernel_size):
                I = inputs[:,m:(m + H - (k - 1)):s, n:(n + W - (k - 1)):s,:]
                K = self._kernel[m, n]
                E = tf.einsum("abcd,de->abce", I, K)
                #print("I shape:", I.shape)
                #print("K shape:", K.shape)
                #print("E shape:", E.shape)
                #print("Output shape:", output.shape)
                output += E

        output = tf.nn.relu(output)


        # If requested, verify that `output` contains a correct value.
        if self._verify:
            reference = tf.nn.relu(tf.nn.convolution(inputs, self._kernel, self._stride) + self._bias)
            np.testing.assert_allclose(output, reference, atol=1e-4, err_msg="Forward pass differs!")

        return output

    def backward(
        self, inputs: tf.Tensor, outputs: tf.Tensor, outputs_gradient: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # TODO: Given this layer's inputs, this layer's outputs,
        # and the gradient with respect to the layer's outputs,
        # compute the derivatives of the loss with respect to
        # - the `inputs` layer,
        # - `self._kernel`,
        # - `self._bias`.
        inputs_gradient, kernel_gradient, bias_gradient = ..., ..., ...

        out_norm = tf.math.divide_no_nan(outputs, tf.math.abs(outputs))
        thru_relu = tf.math.maximum(tf.zeros(out_norm.shape), out_norm)
        G = tf.math.multiply(outputs_gradient, thru_relu)


        #print("outputs:", outputs.shape)
        #print("outputs grad:", outputs_gradient.shape)
        B, o1, o2, o3 = outputs_gradient.shape  # ~G_prime
        bias_gradient = tf.math.reduce_sum(G, [0, 1, 2])

        k = self._kernel_size
        B, H, W, C = inputs.shape
        O = self._filters
        #kernel_gradient = tf.zeros(shape=[B, k, k, C, O])
        kernel_gradient = tf.Variable(tf.initializers.Zeros()([B, k, k, C, O]))  # ~K_prime
        s = self._stride
        #print("H W k s:", H, W, k, s)
        """
        for m in range(k):
            for n in range(k):
                ind = tf.constant([[a, m, n] for a in range(B)])
                for i in range(o1):  # o1
                    for j in range(o2):  # o2
                        I = inputs[:,s*i+m,s*j+n,:]
                        ein = tf.einsum("ab,ac->abc", I, G[:,i,j,:])
                        #print("ein shape:", ein.shape)
                        kernel_gradient = tf.tensor_scatter_nd_add(kernel_gradient, indices=ind, updates=ein)
        """

        for m in range(k):
            for n in range(k):
                ind = tf.constant([[a, m, n] for a in range(B)])
                #for i in range(o1):  # o1
                    #for j in range(o2):  # o2
                I = inputs[:,m:m+s*o1:s,n:n+s*o2:s,:]
                ein = tf.einsum("aefb,aefc->abc", I, G)
                #print("ein shape:", ein.shape)

                kernel_gradient = tf.tensor_scatter_nd_add(kernel_gradient, indices=ind, updates=ein)

        kernel_gradient = tf.reduce_sum(kernel_gradient, axis=0)


        inputs_gradient = tf.Variable(tf.initializers.Zeros()([B, H, W, C]))
        G_pad = tf.pad(G, [[0, 0], [k-1, k-1], [k-1, k-1], [0, 0]])
        print("Kernel:", self._kernel.shape)
        print("G_pad:", G_pad.shape)

        """
        for i_prime in range(H):
            for j_prime in range(W):
                for m_prime in range(k):
                    for n_prime in range(k):
                        #K_prime = kernel_gradient[:, k - 1 - m_prime, k - 1 - n_prime, :, :]
                        #G_prime = G_pad[:, i_prime - (k - 1) + m_prime, j_prime - (k - 1) + n_prime, :]
                        #print(i_prime - (k - 1) + m_prime, j_prime - (k - 1) + n_prime)
                        #print(outputs_gradient.shape)

                        #prod = tf.einsum("abc,ac->ab", self._kernel[:, m_prime, n_prime, :],
                        #                 G_pad[:, i_prime-m_prime, j_prime-n_prime,:])
                        prod = tf.einsum("bc,ac->ab", self._kernel[m_prime, n_prime, :, :],
                                         G_pad[:, i_prime - m_prime + (k - 1), j_prime - n_prime + (k - 1), :])
                        ind2 = tf.constant([[a, i_prime, j_prime] for a in range(B)])
                        inputs_gradient = tf.tensor_scatter_nd_add(inputs_gradient, indices=ind2, updates=prod)
        """

        print(outputs.shape)
        #ind3 = [[a, 0: H: s] for a in range(B)]
        ind3 = []
        for i_0 in range(B):
            for i_1 in range(0, H, s):
                for i_2 in range(0, W, s):
                    for i_3 in range(O):
                        ind3.append([i_0, i_1, i_2, i_3])
        ind3 = tf.constant(ind3)
        #outputs_new = tf.scatter_nd(indices=ind3, updates=outputs, shape=inputs.shape)
        #out_norm_new = tf.math.divide_no_nan(outputs_new, tf.math.abs(outputs_new))
        #thru_relu_new = tf.math.maximum(tf.zeros(out_norm_new.shape), out_norm_new)
        #G_new = tf.math.multiply(outputs_gradient, thru_relu_new)
        #G_pad = tf.pad(G_new, [[0, 0], [k - 1, k - 1], [k - 1, k - 1], [0, 0]])


        for m in range(k):
            for n in range(k):
                #K_prime = kernel_gradient[:, k - 1 - m_prime, k - 1 - n_prime, :, :]
                #G_prime = G_pad[:, i_prime - (k - 1) + m_prime, j_prime - (k - 1) + n_prime, :]
                #print(i_prime - (k - 1) + m_prime, j_prime - (k - 1) + n_prime)
                #print(outputs_gradient.shape)

                #prod = tf.einsum("abc,ac->ab", self._kernel[:, m_prime, n_prime, :],
                #                 G_pad[:, i_prime-m_prime, j_prime-n_prime,:])
                prod = tf.einsum("bc,aefc->aefb", self._kernel[m, n, :, :],
                                 G_pad[:, - m + (k - 1):H - m + (k - 1), - n + (k - 1):W - n + (k - 1), :])
                #ind2 = tf.constant([a for a in range(B)])
                #inputs_gradient = tf.tensor_scatter_nd_add(inputs_gradient, indices=ind2, updates=prod)
                #inputs_gradient = tf.tensor_scatter_nd_add(inputs_gradient, indices=ind2, updates=prod)
                inputs_gradient.assign_add(prod)

        #inputs_gradient = tf.reduce_sum(inputs_gradient, axis=0)

        #bias_gradient = 0
        #kernel_gradient = 0
        #inputs_gradient = 0

        # If requested, verify that the three computed gradients are correct.
        if self._verify:
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                reference = tf.nn.relu(tf.nn.convolution(inputs, self._kernel, self._stride) + self._bias)
            """for name, computed, reference in zip(
                    ["Inputs", "Kernel", "Bias"], [inputs_gradient, kernel_gradient, bias_gradient],
                    tape.gradient(reference, [inputs, self._kernel, self._bias], outputs_gradient)):
                np.testing.assert_allclose(computed, reference, atol=1e-4, err_msg=name + " gradient differs!")"""

            """for name, computed, reference in zip(
                    ["Kernel", "Inputs", "Bias"], [kernel_gradient, inputs_gradient, bias_gradient],
                    tape.gradient(reference, [self._kernel, inputs, self._bias], outputs_gradient)):
                np.testing.assert_allclose(computed, reference, atol=1e-4, err_msg=name + " gradient differs!")"""

            for name, computed, reference in zip(
                    ["Bias", "Kernel", "Inputs"], [bias_gradient, kernel_gradient, inputs_gradient],
                    tape.gradient(reference, [self._bias, self._kernel, inputs], outputs_gradient)):
                np.testing.assert_allclose(computed, reference, atol=1e-4, err_msg=name + " gradient differs!")

        # Return the inputs gradient, the layer variables, and their gradients.
        return inputs_gradient, [self._kernel, self._bias], [kernel_gradient, bias_gradient]


class Model:
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args

        # Create the convolutional layers according to `args.cnn`.
        input_shape = [MNIST.H, MNIST.W, MNIST.C]
        self._convs = []
        for layer in args.cnn.split(","):
            filters, kernel_size, stride = map(int, layer.split("-"))
            self._convs.append(Convolution(filters, kernel_size, stride, input_shape, args.verify))
            input_shape = [(input_shape[0] - kernel_size) // stride + 1,
                           (input_shape[1] - kernel_size) // stride + 1, filters]

        # Create the classification head
        self._flatten = tf.keras.layers.Flatten(input_shape=input_shape)
        self._classifier = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)

        # Create the metric and the optimizer
        self._accuracy = tf.metrics.SparseCategoricalAccuracy()
        self._optimizer = tf.optimizers.Adam(args.learning_rate, jit_compile=False)

    def train_epoch(self, dataset: MNIST.Dataset) -> None:
        for batch in dataset.batches(self._args.batch_size):
            # Forward pass through the convolutions
            hidden = tf.constant(batch["images"])
            conv_values = [hidden]
            for conv in self._convs:
                hidden = conv.forward(hidden)
                conv_values.append(hidden)

            # Run the classification head
            hidden_flat = self._flatten(hidden)
            predictions = self._classifier(hidden_flat)

            # Compute the gradients of the classifier and the convolution output
            d_logits = (predictions - tf.one_hot(batch["labels"], MNIST.LABELS)) / len(batch["images"])
            variables = [self._classifier.bias, self._classifier.kernel]
            gradients = [tf.reduce_sum(d_logits, 0), tf.linalg.matmul(hidden_flat, d_logits, transpose_a=True)]
            hidden_gradient = tf.reshape(tf.linalg.matvec(self._classifier.kernel, d_logits), hidden.shape)

            # Backpropagate the gradient through the convolutions
            for conv, inputs, outputs in reversed(list(zip(self._convs, conv_values[:-1], conv_values[1:]))):
                hidden_gradient, conv_variables, conv_gradients = conv.backward(inputs, outputs, hidden_gradient)
                variables.extend(conv_variables)
                gradients.extend(conv_gradients)

            # Update the weights
            self._optimizer.apply_gradients(zip(gradients, variables))

    def evaluate(self, dataset: MNIST.Dataset) -> float:
        self._accuracy.reset_states()
        for batch in dataset.batches(self._args.batch_size):
            hidden = batch["images"]
            for conv in self._convs:
                hidden = conv.forward(hidden)
            hidden = self._flatten(hidden)
            predictions = self._classifier(hidden)
            self._accuracy(batch["labels"], predictions)
        return self._accuracy.result()


def main(args: argparse.Namespace) -> float:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    # Load data, using only 5 000 training images
    mnist = MNIST(size={"train": 5_000})

    # Create the model
    model = Model(args)

    for epoch in range(args.epochs):
        model.train_epoch(mnist.train)

        dev_accuracy = model.evaluate(mnist.dev)
        print("Dev accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * dev_accuracy))

    test_accuracy = model.evaluate(mnist.test)
    print("Test accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * test_accuracy))

    # Return dev and test accuracies for ReCodEx to validate.
    return dev_accuracy, test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
