#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import Dict
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

# Pepa: 3d76595a-e687-11e9-9ce9-00505601122b
# Nekdo: 594215cf-e687-11e9-9ce9-00505601122b

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")  # 5
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        # Create a model with two inputs, both images of size [MNIST.H, MNIST.W, MNIST.C].
        images = (
            tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C]),
            tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C]),
        )

        # TODO: The model starts by passing each input image through the same
        # subnetwork (with shared weights), which should perform
        # - convolution with 10 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation,
        # - convolution with 20 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation,
        # - flattening layer,
        # - fully connected layer with 200 neurons and ReLU activation,
        # obtaining a 200-dimensional feature vector FV of each image.
        conv1 = tf.keras.layers.Conv2D(10, 3, strides=2, padding="valid", activation=tf.nn.relu)  # maybe Conv1D?, use_bias=False?
        conv2 = tf.keras.layers.Conv2D(20, 3, strides=2, padding="valid", activation=tf.nn.relu)  # maybe Conv1D?, use_bias=False?
        flatten = tf.keras.layers.Flatten()
        dense = tf.keras.layers.Dense(200, activation=tf.nn.relu)  # use_bias=False?

        inputs_1 = images[0]
        hidden_1 = conv1(inputs_1)
        hidden_1 = conv2(hidden_1)
        hidden_1 = flatten(hidden_1)
        hidden_1 = dense(hidden_1)

        inputs_2 = images[1]
        hidden_2 = conv1(inputs_2)
        hidden_2 = conv2(hidden_2)
        hidden_2 = flatten(hidden_2)
        hidden_2 = dense(hidden_2)

        # TODO: Using the computed representations, the model should produce four outputs:
        # - first, compute _direct comparison_ whether the first digit is
        #   greater than the second, by
        #   - concatenating the two 200-dimensional image representations FV,
        #   - processing them using another 200-neuron ReLU dense layer
        #   - computing one output using a dense layer with `tf.nn.sigmoid` activation
        # - then, classify the computed representation FV of the first image using
        #   a densely connected softmax layer into 10 classes;
        # - then, classify the computed representation FV of the second image using
        #   the same layer (identical, i.e., with shared weights) into 10 classes;
        # - finally, compute _indirect comparison_ whether the first digit
        #   is greater than second, by comparing the predictions from the above
        #   two outputs.
        outputs = {
            "direct_comparison": ...,
            "digit_1": ...,
            "digit_2": ...,
            "indirect_comparison": ...,
        }

        concatted = tf.keras.layers.Concatenate()([hidden_1, hidden_2])
        dense2 = tf.keras.layers.Dense(200, activation=tf.nn.relu)  # use_bias=False?
        output_both = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)  # use_bias=False?
        hid_both = dense2(concatted)
        hid_both = output_both(hid_both)
        outputs["direct_comparison"] = hid_both

        output_solo = tf.keras.layers.Dense(10, activation=tf.nn.softmax)  # use_bias=False?
        outputs["digit_1"] = output_solo(hidden_1)
        outputs["digit_2"] = output_solo(hidden_2)

        outputs["indirect_comparison"] = (tf.argmax(outputs["digit_1"], axis=1) > tf.argmax(outputs["digit_2"], axis=1))

        # Finally, construct the model.
        super().__init__(inputs=images, outputs=outputs)

        # Note that for historical reasons, names of a functional model outputs
        # (used for displayed losses/metric names) are derived from the name of
        # the last layer of the corresponding output. Here we instead use
        # the keys of the `outputs` dictionary.
        self.output_names = sorted(outputs.keys())

        # TODO: Define the appropriate losses for the model outputs
        # "direct_comparison", "digit_1", "digit_2". Regarding metrics,
        # the accuracy of both the direct and indirect comparisons should be
        # computed; name both metrics "accuracy" (i.e., pass "accuracy" as the
        # first argument of the metric object).
        self.compile(
            optimizer=tf.keras.optimizers.Adam(jit_compile=False),
            loss={
                #"direct_comparison": ...,
                #"digit_1": ...,
                #"digit_2": ...,
                "digit_1": tf.losses.SparseCategoricalCrossentropy(),
                "digit_2": tf.losses.SparseCategoricalCrossentropy(),
                "direct_comparison": tf.losses.BinaryCrossentropy(),
            },
            metrics={
                #"direct_comparison": [...],
                #"indirect_comparison": [...],
                "direct_comparison": [tf.metrics.BinaryAccuracy(name="accuracy")],
                "indirect_comparison": [tf.metrics.BinaryAccuracy(name="accuracy")],
            },
        )
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    # Create an appropriate dataset using the MNIST data.
    def create_dataset(
        self, mnist_dataset: MNIST.Dataset, args: argparse.Namespace, training: bool = False
    ) -> tf.data.Dataset:
        # Start by using the original MNIST data
        dataset = tf.data.Dataset.from_tensor_slices((mnist_dataset.data["images"], mnist_dataset.data["labels"]))

        # TODO: If `training`, shuffle the data with `buffer_size=10_000` and `seed=args.seed`.
        if training:
            dataset = dataset.shuffle(buffer_size=10_000, seed=args.seed)

        # TODO: Combine pairs of examples by creating batches of size exactly 2 (you would throw
        # away the last example if the original dataset size were odd; but in MNIST it is even).
        dataset = dataset.batch(2)  # batch_size=2

        # TODO: Map pairs of images to elements suitable for our model. Notably,
        # the elements should be pairs `(input, output)`, with
        # - `input` being a pair of images,
        # - `output` being a dictionary with keys "digit_1", "digit_2", "direct_comparison",
        #   and "indirect_comparison".
        def create_element(images, labels):
            #...
            inp = (images[0], images[1])
            if labels[0] <= labels[1]:
                lab = 0
            else:
                lab = 1
            out = {
                "digit_1": labels[0], "digit_2": labels[1],
                "direct_comparison": lab,
                "indirect_comparison": lab,
                   }
            return inp, out
        dataset = dataset.map(create_element)

        # TODO: Create batches of size `args.batch_size`
        dataset = dataset.batch(args.batch_size)  # batch_size=args.batch_size

        return dataset


def main(args: argparse.Namespace) -> Dict[str, float]:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data
    mnist = MNIST()

    # Create the model
    model = Model(args)

    # Construct suitable datasets from the MNIST data.
    train = model.create_dataset(mnist.train, args, training=True)
    dev = model.create_dataset(mnist.dev, args)

    # Train
    logs = model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[model.tb_callback])

    # Return development metrics for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
