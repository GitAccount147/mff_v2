#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from modelnet import ModelNet

# Shrek and Shrek:
# 3d76595a-e687-11e9-9ce9-00505601122b
# 594215cf-e687-11e9-9ce9-00505601122b

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--modelnet", default=32, type=int, help="ModelNet dimension.")  # 20 or 32
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

# our:
# TO-DO: label smoothing; ensemble; weight_decay cosine; residual layer
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay.")
parser.add_argument("--cnn", default="CB-8-3-1-same,CB-8-3-1-same,M-3-2,CB-16-3-1-same,CB-16-3-1-same,M-3-2,F,H-100",
                    type=str, help="CNN architecture.")
# "CB-8-3-1-same,CB-8-3-1-same,M-3-2,CB-16-3-1-same,CB-16-3-1-same,M-3-2,F,H-100",
# "CB-8-3-5-valid,R-[CB-8-3-1-same,CB-8-3-1-same],F,H-50",
# "CB-8-3-5-valid,R-[CB-8-3-1-same,CB-8-3-1-same],F,H-100",
# "CB-8-3-5-valid,R-[CB-8-3-1-same,CB-8-3-1-same],F,H-500",
# "CB-8-3-1-same,CB-8-3-1-same,M-3-2,CB-16-3-1-same,CB-16-3-1-same,M-3-2,F,H-100",
# "CB-8-3-1-same,CB-8-3-1-same,M-3-2,CB-16-3-1-same,CB-16-3-1-same,M-3-2,F,H-500",
# "CB-8-3-1-same,CB-8-3-1-same,M-3-2,CB-16-3-1-same,CB-16-3-1-same,M-3-2,F,H-500,D-0.5",
# "CB-8-3-1-same,CB-8-3-1-same,M-3-2,CB-16-3-1-same,CB-16-3-1-same,M-3-2,F,H-1000,D-0.5",
# "CB-8-3-5-valid,R-[CB-8-3-1-same,CB-8-3-1-same],F,H-500,D-0.5",
# "C-32-3-1-same,C-32-3-1-same,M-2-2,D-0.25,C-64-3-1-same,C-64-3-1-same,M-2-2,D-0.25,F,H-512,D-0.5",
# "F,H-500,D-0.5",
# "CB-32-3-1-same,CB-32-3-1-same,M-2-2,D-0.25,CB-64-3-1-same,CB-64-3-1-same,M-2-2,D-0.25,F,H-512,D-0.5",
# "CB-32-3-1-same,CB-32-3-1-same,M-2-2,D-0.2,CB-64-3-1-same,CB-64-3-1-same,M-2-2,D-0.3,CB-128-3-1-same,CB-128-3-1-same,M-2-2,D-0.4,F,H-128,D-0.5",
# "CB-32-3-1-same,CB-32-3-1-same,M-2-2,D-0.1,CB-64-3-1-same,CB-64-3-1-same,M-2-2,D-0.2,CB-128-3-1-same,CB-128-3-1-same,M-2-2,D-0.3,CB-256-3-1-same,CB-256-3-1-same,M-2-2,D-0.4,F,H-64,D-0.5",
# "CB-32-3-1-same,CB-32-3-1-same,M-2-2,D-0.2,CB-64-3-1-same,CB-64-3-1-same,M-2-2,D-0.3,CB-128-3-1-same,CB-128-3-1-same,M-2-2,D-0.4,F,H-512,D-0.5",
# "R-[CB-32-3-1-same,CB-32-3-1-same],M-2-2,D-0.25,CB-64-3-1-same,CB-64-3-1-same,M-2-2,D-0.25,F,H-512,D-0.5"

class ToiletRecogniser(tf.keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        #inputs = tf.keras.layers.Input(shape=[ModelNet.D, ModelNet.H, ModelNet.W, ModelNet.C])
        inputs = tf.keras.layers.Input(shape=[args.modelnet, args.modelnet, args.modelnet, ModelNet.C])
        hidden = inputs

        """
        conv = tf.keras.layers.Conv3D(filters=8, kernel_size=1, strides=1, padding="valid", use_bias=False)
        batch_norm = tf.keras.layers.BatchNormalization()
        max_pool = tf.keras.layers.MaxPool3D(pool_size=2, strides=2)
        flatten = tf.keras.layers.Flatten()
        dense = tf.keras.layers.Dense(units=512)
        dropout = tf.keras.layers.Dropout(rate=0.5)

        hidden = conv(inputs)
        hidden = batch_norm(hidden)
        hidden = max_pool(hidden)
        hidden = flatten(hidden)
        hidden = dense(hidden)
        hidden = dropout(hidden)
        """


        layers = args.cnn.split(',')
        for i in range(len(layers)):
            params = layers[i].split("-")
            layer_type = params[0]
            layer = ...
            if layer_type == "CB":  # 3D convolution with batch normalization
                conv = tf.keras.layers.Conv3D(filters=int(params[1]), kernel_size=int(params[2]),
                                              strides=int(params[3]), padding=params[4], use_bias=False)
                bn = tf.keras.layers.BatchNormalization()
                activation = tf.nn.relu
                hidden = activation(bn(conv(hidden)))
            elif layer_type == "C":  # 3D convolution
                layer = tf.keras.layers.Conv3D(filters=int(params[1]), kernel_size=int(params[2]),
                                                 strides=int(params[3]), padding=params[4], activation=tf.nn.relu)
            elif params[0] == 'M':  # max_pool
                layer = tf.keras.layers.MaxPool3D(pool_size=int(params[1]), strides=int(params[2]))
            elif params[0] == 'F':  # flatten
                layer = tf.keras.layers.Flatten()  # input size?
            elif params[0] == 'H':  # dense
                layer = tf.keras.layers.Dense(int(params[1]), activation=tf.nn.relu)
            elif params[0] == 'D':  # dropout
                layer = tf.keras.layers.Dropout(rate=float(params[1]))

            if layer_type != "CB":
                hidden = layer(hidden)

        # Add the final output layer
        outputs = tf.keras.layers.Dense(len(ModelNet.LABELS), activation=tf.nn.softmax)(hidden)

        """
        hidden = ...
        layers = cnn_arch.split(',')
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
        outputs = tf.keras.layers.Dense(len(CAGS.LABELS), activation=tf.nn.softmax)(hidden)
        """


        super().__init__(inputs=inputs, outputs=outputs)
        self.compile(
            optimizer=tf.optimizers.Adam(jit_compile=False, learning_rate=args.learning_rate,
                                         weight_decay=args.weight_decay),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)


def main(args: argparse.Namespace) -> None:
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

    # Load the data
    modelnet = ModelNet(args.modelnet)

    # TODO: Create the model and train it
    #model = ...
    model = ToiletRecogniser(args=args)

    train_data = modelnet.train.data["voxels"]
    train_labels = modelnet.train.data["labels"]
    dev_data = modelnet.dev.data["voxels"]
    dev_labels = modelnet.dev.data["labels"]
    test_data = modelnet.test.data["voxels"]


    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1)  # maybe delete

    logs = model.fit(
        train_data, train_labels,
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(dev_data, dev_labels),
        callbacks=[tb_callback],  # maybe delete
    )


    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "3d_recognition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the probabilities on the test set
        #test_probabilities = model.predict(...)
        test_probabilities = model.predict(test_data)

        for probs in test_probabilities:
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
