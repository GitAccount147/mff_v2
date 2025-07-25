#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import Dict, Iterable, Optional, Tuple, Union
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from omniglot_dataset import Omniglot

# 3d76595a-e687-11e9-9ce9-00505601122b
# 594215cf-e687-11e9-9ce9-00505601122b

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--cell_size", default=40, type=int, help="Memory cell size")
parser.add_argument("--classes", default=2, type=int, help="Number of classes per episode.")  # 5
parser.add_argument("--debug", default=True, action="store_true", help="If given, run functions eagerly.")  # False
parser.add_argument("--epochs", default=3, type=int, help="Number of epochs.")  # 20
parser.add_argument("--images_per_class", default=10, type=int, help="Images per class.")
parser.add_argument("--lstm_dim", default=256, type=int, help="LSTM Dim")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--read_heads", default=1, type=int, help="Read heads.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--test_episodes", default=160, type=int, help="Number of testing episodes.")  # 1_000
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--train_episodes", default=160, type=int, help="Number of training episodes.")  # 10_000
# If you add more arguments, ReCodEx will keep them with your default values.


class EpisodeGenerator():
    """Python generator of episodes."""
    def __init__(self, dataset: Omniglot.Dataset, args: argparse.Namespace, seed: int) -> None:
        self._dataset = dataset
        self._args = args

        # Random generator
        self._generator = np.random.RandomState(seed)

        # Create required indexes
        self._unique_labels = np.unique(dataset.data["labels"])
        self._label_indices = {}
        for i, label in enumerate(dataset.data["labels"]):
            self._label_indices.setdefault(label, []).append(i)

    def __call__(self) -> Iterable[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]]:
        """Generate infinite number of episodes.

        Every episode contains `self._args.classes` randomly chosen Omniglot
        classes, each class being assigned a randomly chosen label. For every
        chosen class, `self._args.images_per_class` images are randomly selected.

        Apart from the images, the input contains the random labels one step
        after the corresponding images (with the first label being -1).
        The gold outputs are also the labels, but without the one-step offset.
        """
        while True:
            indices, labels = [], []
            for index, label in enumerate(self._generator.choice(
                    self._unique_labels, size=self._args.classes, replace=False)):
                indices.extend(self._generator.choice(
                    self._label_indices[label], size=self._args.images_per_class, replace=False))
                labels.extend([index] * self._args.images_per_class)
            indices, labels = np.array(indices, np.int32), np.array(labels, np.int32)

            permutation = self._generator.permutation(len(indices))
            images = self._dataset.data["images"][indices[permutation]]
            labels = labels[permutation]
            yield (images, np.pad(labels[:-1], [[1, 0]], constant_values=-1)), labels


class Model(tf.keras.Model):
    class NthOccurenceAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):
        """A sparse categorical accuracy computed only for `nth` occurrence of every element."""
        def __init__(self, nth: int, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._nth = nth

        def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: Optional[tf.Tensor]) -> None:
            assert sample_weight is None
            one_hot = tf.one_hot(y_true, tf.reduce_max(y_true) + 1)
            nth = tf.math.reduce_sum(tf.math.cumsum(one_hot, axis=-2) * one_hot, axis=-1)
            indices = tf.where(nth == self._nth)
            return super().update_state(tf.gather_nd(y_true, indices), tf.gather_nd(y_pred, indices))

    class MemoryAugmentedLSTMCell(tf.keras.layers.AbstractRNNCell):
        """The LSTM controller augmented with external memory.

        The LSTM has dimensionality `units`. The external memory consists
        of `memory_cells` cells, each being a vector of `cell_size` elements.
        The controller has `read_heads` read heads and one write head.
        """
        def __init__(self, units: int, memory_cells: int, cell_size: int, read_heads: int, **kwargs) -> None:
            super().__init__(**kwargs)
            self._memory_cells = memory_cells
            self._cell_size = cell_size
            self._read_heads = read_heads

            # TODO: Create the required layers:
            # - `self._controller` is a `tf.keras.layers.LSTMCell` with `units` units;
            # - `self._parameters` is a `tanh`-activated dense layer with `(read_heads + 1) * cell_size` units;
            # - `self._output_layer` is a `tanh`-activated dense layer with `units` units.
            #self._controller = ...
            #self._parameters = ...
            #self._output_layer = ...
            self._controller = tf.keras.layers.LSTMCell(units=units)
            self._parameters = tf.keras.layers.Dense(units=((read_heads + 1) * cell_size), activation=tf.nn.tanh)
            self._output_layer = tf.keras.layers.Dense(units=units, activation=tf.nn.tanh)

        @property
        def state_size(self) -> Tuple[Union[int, tf.TensorShape]]:
            # TODO: Return the description of the state size as a (possibly nested)
            # tuple (or a list) containing shapes of individual state tensors. The state
            # of the `MemoryAugmentedLSTMCell` consists of the following components:
            # - first the state tensors of the `self._controller` itself; note that
            #   the `self._controller` also has `state_size` property;
            # - then the values of memory cells read by the `self._read_heads` heads
            #   in the previous time step;
            # - finally the external memory itself, which is a matrix containing
            #   `self._memory_cells` cells as rows, each of length `self._cell_size`.
            # A tensor shape is specified without the batch size, either as:
            # - an integer, in which case the state tensor for a single example
            #   is a vector of the given size; or
            # - a `tf.TensorShape`, which allows declaring tensors of different
            #   dimensionality than just vectors.
            #raise NotImplementedError()
            controller_states = self._controller.state_size
            memory_cells = self._read_heads * self._cell_size
            external_memory = tf.TensorShape([self._memory_cells, self._cell_size])
            result = controller_states, memory_cells, external_memory
            return result


        def call(self, inputs: tf.Tensor, states: Tuple[tf.Tensor]) -> Tuple[tf.Tensor, Tuple[tf.Tensor]]:
            # TODO: Decompose `states` into `controller_state`, `read_value` and `memory`
            # (see `state_size` describing the `states` structure).
            #controller_state, read_value, memory = ...
            controller_state = states[0]
            read_value = states[1]
            memory = states[2]

            # TODO: Call the LSTM controller, using a concatenation of `inputs` and
            # `read_value` (in this order) as input and `controller_state` as state.
            # Store the results in `controller_output` and `controller_state`.
            #controller_output, controller_state = ...
            concat = tf.concat([inputs, read_value], axis=-1)
            controller_output, controller_state = self._controller(inputs=concat, states=controller_state)

            # TODO: Pass the `controller_output` through the `self._parameters` layer, obtaining
            # the parameters for interacting with the external memory (in this order):
            # - `write_value` is the first `self._cell_size` elements of every batch example;
            # - `read_keys` is the rest of the elements of every batch example, reshaped to
            #   `[batch_size, self._read_heads, self._cell_size]`.
            #write_value = ...
            #read_keys = ...
            batch_size = tf.shape(inputs)[0]
            params_for_external = self._parameters(controller_output)

            write_value = params_for_external[:, :self._cell_size]
            read_keys_raw = params_for_external[:, self._cell_size:]

            read_keys = tf.reshape(tensor=read_keys_raw, shape=[batch_size, self._read_heads, self._cell_size])

            # TODO: Read the memory. For every predicted read key, the goal is to
            # - compute cosine similarities between the key and all memory cells;
            # - compute cell distribution as a softmax of the computed cosine similarities;
            # - the read value is the sum of the memory cells weighted by the above distribution.
            #
            # However, implement the reading process in a vectorized way (for all read keys in parallel):
            # - compute L2 normalized copy of `memory` and `read_keys`, using `tf.math.l2_normalize`,
            #   so that every cell vector has norm 1;
            # - compute the self-attention between the L2-normalized copy of `read_keys` and `memory`
            #   with a single matrix multiplication, obtaining a tensor with shape
            #   `[batch_size, self._read_heads, self._memory_cells]`. You will need to transpose one
            #   of the matrices -- do not transpose it manually, but use `tf.linalg.matmul` capable of
            #   transposing the matrices to be multiplied (see parameters `transpose_a` and `transpose_b`).
            # - apply softmax, resulting in a distribution over the memory cells for every read key
            # - compute weighted sum of the original (non-L2-normalized) `memory` according to the
            #   obtained distribution. Compute it using a single matrix multiplication, producing
            #   a value with shape `[batch_size, self._read_heads, self._cell_size]`.
            # Finally, reshape the result into `read_value` of shape `[batch_size, self._read_heads * self._cell_size]`
            #read_value = ...
            memory_normalized = tf.math.l2_normalize(memory, axis=-1)
            read_keys_normalized = tf.math.l2_normalize(read_keys, axis=-1)
            multiplied = tf.linalg.matmul(read_keys_normalized, memory_normalized, transpose_b=True)

            soft = tf.nn.softmax(multiplied)
            weighted_sum = tf.linalg.matmul(soft, memory)

            read_value = tf.reshape(tensor=weighted_sum, shape=[batch_size, self._read_heads * self._cell_size])

            # TODO: Write to the memory by prepending the `write_value` as the first cell (row);
            # the last memory cell (row) is dropped.
            #memory = ...
            write_value_expanded = tf.expand_dims(write_value, axis=1)
            memory = tf.concat([write_value_expanded, memory[:, :-1, :]], axis=1)

            # TODO: Generate `output` by concatenating `controller_output` and `read_value`
            # (in this order) and passing it through the `self._output_layer`.
            #output = ...
            concat = tf.concat([controller_output, read_value], axis=-1)
            output = self._output_layer(concat)

            # TODO: Return the `output` as output and a suitable combination of
            # `controller_state`, `read_value` and `memory` as state.
            #raise NotImplementedError()
            state = (controller_state, read_value, memory)
            return output, state

    def __init__(self, args: argparse.Namespace) -> None:
        # Construct the model. The inputs are:
        # - a sequence of `images`;
        # - a sequence of labels of the previous images.
        images = tf.keras.layers.Input([None, Omniglot.H, Omniglot.W, Omniglot.C], dtype=tf.float32)
        previous_labels = tf.keras.layers.Input([None], dtype=tf.int32)

        # TODO: Process each image with the same sequence of the following operations:
        # - apply the `tf.keras.layers.Rescaling(1/255.)` layer to scale the images to [0, 1] range;
        # - convolutional layer with 8 filters, 3x3 kernel, stride 2, valid padding; BatchNorm; ReLU;
        # - convolutional layer with 16 filters, 3x3 kernel, stride 2, valid padding; BatchNorm; ReLU;
        # - convolutional layer with 32 filters, 3x3 kernel, stride 2, valid padding; BatchNorm; ReLU;
        # - finally, flatten each image into a vector.
        # Do not forget about `use_bias=False` in every convolution before batch normalization.
        # The batch normalization should be computed across all the images.
        rescale = tf.keras.layers.Rescaling(1/255.)
        conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=2, padding="valid", use_bias=False)
        conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=2, padding="valid", use_bias=False)
        conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="valid", use_bias=False)
        bn1 = tf.keras.layers.BatchNormalization()  # do we need multiple instances?
        bn2 = tf.keras.layers.BatchNormalization()
        bn3 = tf.keras.layers.BatchNormalization()
        relu = tf.nn.relu

        hidden = rescale(images)
        hidden = relu(bn1(conv1(hidden)))
        hidden = relu(bn2(conv2(hidden)))
        hidden = relu(bn3(conv3(hidden)))

        batch_size, number_of_images = tf.shape(hidden)[0], tf.shape(hidden)[1]
        hidden = tf.reshape(hidden, shape=[batch_size, number_of_images, 128])

        # TODO: To create the input for the `MemoryAugmentedLSTMCell`, concatenate (in this order)
        # each computed image representation with the one-hot representation (with `args.classes` classes)
        # of the label of the previous image from `previous_labels`.

        one_hot = tf.one_hot(previous_labels, args.classes)
        concat = tf.concat([hidden, one_hot], axis=2)
        print("concat:", concat)

        # TODO: Create the `MemoryAugmentedLSTMCell` cell, using
        # - `args.lstm_dim` units;
        # - `args.classes * args.images_per_class` memory cells of size `args.cell_size`;
        # - `args.read_heads` read heads.
        # Then, run this cell using `tf.keras.layers.RNN` on the prepared input,
        # obtaining output for every input sequence element.
        MALC = Model.MemoryAugmentedLSTMCell(units=args.lstm_dim, memory_cells=(args.classes * args.images_per_class),
                                             cell_size=args.cell_size, read_heads=args.read_heads)
        rnn = tf.keras.layers.RNN(cell=MALC, return_sequences=True)
        hidden = rnn(concat)

        # TODO: Pass the sequence of outputs through a classification dense layer
        # with `args.classes` units and `tf.nn.softmax` activation.
        #predictions = ...
        dense = tf.keras.layers.Dense(units=args.classes, activation=tf.nn.softmax)
        predictions = dense(hidden)

        # Create the model and compile it.
        super().__init__(inputs=[images, previous_labels], outputs=predictions)
        self.compile(
            optimizer=tf.optimizers.Adam(jit_compile=False),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="acc"),
                     *[self.NthOccurenceAccuracy(i, name="acc{}".format(i)) for i in [1, 2, 5, 10]]],
        )


def main(args: argparse.Namespace) -> Dict[str, float]:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        # tf.data.experimental.enable_debug_mode()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data. The images contain a single channel of values
    # of type `tf.uint8` in [0-255] range.
    omniglot = Omniglot()

    def create_dataset(data: Omniglot.Dataset, seed: int) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_generator(
            EpisodeGenerator(data, args, seed=seed),
            output_signature=(
                (tf.TensorSpec([None, Omniglot.H, Omniglot.W, Omniglot.C], tf.uint8),
                 tf.TensorSpec([None], tf.int32)),
                tf.TensorSpec([None], tf.int32),
            )
        )
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(tf.data.INFINITE_CARDINALITY))
        return dataset
    train = create_dataset(omniglot.train, args.seed).take(args.train_episodes).batch(args.batch_size).prefetch(1)
    test = create_dataset(omniglot.test, seed=42).take(args.test_episodes).batch(args.batch_size).cache()

    # Create the model and train
    model = Model(args)
    logs = model.fit(train, epochs=args.epochs, validation_data=test)

    # Return the training and development losses for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items() if "loss" in metric}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
