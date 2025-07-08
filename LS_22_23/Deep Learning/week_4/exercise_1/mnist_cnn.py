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
# Anonymous: 594215cf-e687-11e9-9ce9-00505601122b

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--cnn", default="CB-8-3-5-valid,R-[CB-8-3-1-same,CB-8-3-1-same],F,H-50", type=str, help="CNN architecture.")  # None
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")  # 10
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        # TODO: Create the model. The template uses the functional API, but
        # feel free to use subclassing if you want.
        inputs = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])

        # TODO: Add CNN layers specified by `args.cnn`, which contains
        # a comma-separated list of the following layers:
        # - `C-filters-kernel_size-stride-padding`: Add a convolutional layer with ReLU
        #   activation and specified number of filters, kernel size, stride and padding.
        # - `CB-filters-kernel_size-stride-padding`: Same as `C`, but use batch normalization.
        #   In detail, start with a convolutional layer without bias and activation,
        #   then add a batch normalization layer, and finally the ReLU activation.
        # - `M-pool_size-stride`: Add max pooling with specified size and stride, using
        #   the default "valid" padding.
        # - `R-[layers]`: Add a residual connection. The `layers` contain a specification
        #   of at least one convolutional layer (but not a recursive residual connection `R`).
        #   The input to the `R` layer should be processed sequentially by `layers`, and the
        #   produced output (after the ReLU nonlinearity of the last layer) should be added
        #   to the input (of this `R` layer).
        # - `F`: Flatten inputs. Must appear exactly once in the architecture.
        # - `H-hidden_layer_size`: Add a dense layer with ReLU activation and the specified size.
        # - `D-dropout_rate`: Apply dropout with the given dropout rate.
        # You can assume the resulting network is valid; it is fine to crash if it is not.
        #
        # Produce the results in the variable `hidden`.
        hidden = ...

        # hidden is not sequential but functional
        # (via labs 3 on GitHub (https://github.com/ufal/npfl114/blob/master/labs/03/example_keras_models.py))
        layers = args.cnn.split(',')
        layer_end = [inputs]  # all sequential layers (always connecting the new one and the last (=layer_end[-1]))
        i = 0
        while i < len(layers):
            if layers[i][0] != 'R':
                params = layers[i].split('-')
                if len(params[0]) == 2:  # 'CB' convolution with batch_norm
                    conv = tf.keras.layers.Conv2D(int(params[1]), int(params[2]), strides=int(params[3]),
                                                  padding=params[4], use_bias=False)
                    bn = tf.keras.layers.BatchNormalization()
                    activation = tf.nn.relu
                    hidden = activation(bn(conv(layer_end[-1])))
                else:
                    this_layer = ...
                    if params[0] == 'C':  # 'C'  convolution
                        this_layer = tf.keras.layers.Conv2D(int(params[1]), int(params[2]), strides=int(params[3]),
                                                      padding=params[4], activation=tf.nn.relu)
                    elif params[0] == 'M':  # 'M' max_pool
                        this_layer = tf.keras.layers.MaxPool2D(pool_size=int(params[1]), strides=int(params[2]))
                    elif params[0] == 'F':  # 'F' flatten
                        this_layer = tf.keras.layers.Flatten()  # input size?
                    elif params[0] == 'H':  # 'H' dense
                        this_layer = tf.keras.layers.Dense(int(params[1]), activation=tf.nn.relu)
                    elif params[0] == 'D':  # 'D' dropout
                        this_layer = tf.keras.layers.Dropout(rate=float(params[1]))
                    hidden = this_layer(layer_end[-1])
                layer_end.append(hidden)
            else:  # 'R' residual
                # remember from which layer we split the residual (-> layer_end[R_start] + last_residual_layer):
                R_start = i
                R_finished = False
                while not R_finished:
                    if layers[i][2] == '[':  # first layer after residual
                        current_layer = layers[i][3:]
                    elif layers[i][-1] == ']':  # last layer of residual
                        current_layer = layers[i][:-1]
                        R_finished = True
                        i -= 1
                    else:
                        current_layer = layers[i]

                    params = current_layer.split('-')

                    # (copied from above):
                    if len(params[0]) == 2:  # 'CB' convolution with batch_norm
                        conv = tf.keras.layers.Conv2D(int(params[1]), int(params[2]), strides=int(params[3]),
                                                      padding=params[4], use_bias=False)
                        bn = tf.keras.layers.BatchNormalization()
                        activation = tf.nn.relu
                        hidden = activation(bn(conv(layer_end[-1])))
                    else:  # 'C'  convolution
                        conv = tf.keras.layers.Conv2D(int(params[1]), int(params[2]), strides=int(params[3]),
                                                      padding=params[4], activation=tf.nn.relu)
                        hidden = conv(layer_end[-1])
                    layer_end.append(hidden)
                    i += 1
                hidden = tf.keras.layers.add([layer_end[R_start], layer_end[-1]])
                layer_end.append(hidden)
            i += 1

        # Add the final output layer
        outputs = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)(hidden)

        super().__init__(inputs=inputs, outputs=outputs)
        self.compile(
            optimizer=tf.optimizers.Adam(jit_compile=False),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)


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

    # Create the model and train it
    model = Model(args)

    logs = model.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        #callbacks=[model.tb_callback],
    )

    # Return development metrics for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
