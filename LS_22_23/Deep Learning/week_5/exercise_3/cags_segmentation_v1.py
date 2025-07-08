#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from cags_dataset import CAGS

# Supr Dupr Tym:
# 3d76595a-e687-11e9-9ce9-00505601122b
# 594215cf-e687-11e9-9ce9-00505601122b

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of the threads 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=5, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

# our hyperparameters:
parser.add_argument("--cnn", default="CB-8-3-1-same,CB-8-3-1-same,M-3-2,CB-16-3-1-same,CB-16-3-1-same,M-3-2,F,H-100", type=str, help="CNN architecture.")

class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace, dataset, back) -> None:
        #inputs = tf.keras.layers.Input(shape=[dataset.H, dataset.W, dataset.C])
        back.trainable = False
        inputs = back.inputs

        top_activation, block5e_add, block3b_add, block2b_add, block1a_project_activation = back(inputs)
        hidden = tf.keras.layers.Conv2DTranspose(filters=1280, kernel_size=2, strides=2)(block1a_project_activation)
        hidden = tf.nn.relu(hidden)


        """
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
        """

        # Add the final output layer
        #outputs = tf.keras.layers.Dense([dataset.H, dataset.W, dataset.C], activation=tf.nn.softmax)(hidden)
        hidden = tf.keras.layers.Conv2D(filters=1280, kernel_size=1)(hidden)
        outputs = tf.nn.sigmoid(hidden)

        super().__init__(inputs=inputs, outputs=outputs)
        self.compile(
            optimizer=tf.optimizers.Adam(jit_compile=False),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)


def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data. Note that both the "image" and the "mask" images
    # are represented using `tf.uint8`s in [0-255] range.
    cags = CAGS()

    # Load the EfficientNetV2-B0 model. It assumes the input images are
    # represented in [0-255] range using either `tf.uint8` or `tf.float32` type.
    backbone = tf.keras.applications.EfficientNetV2B0(include_top=False)

    # Extract features of different resolution. Assuming 224x224 input images
    # (you can set this explicitly via `input_shape` of the above constructor),
    # the below model returns five outputs with resolution 7x7, 14x14, 28x28, 56x56, 112x112.
    backbone = tf.keras.Model(
        inputs=backbone.input,
        outputs=[backbone.get_layer(layer).output for layer in [
             "top_activation", "block5e_add", "block3b_add", "block2b_add", "block1a_project_activation"]]
    )

    # TODO: Create the model and train it
    #model = ...

    train_dataset = tf.data.TFRecordDataset("cags.train.tfrecord")
    train_dataset = train_dataset.map(CAGS.parse)
    train_dataset = train_dataset.map(lambda example: (example["image"], example["mask"]))
    train_dataset = train_dataset.batch(args.batch_size)
    dev_dataset = tf.data.TFRecordDataset("cags.dev.tfrecord")
    dev_dataset = dev_dataset.map(CAGS.parse)
    dev_dataset = dev_dataset.map(lambda example: (example["image"], example["mask"]))
    dev_dataset = dev_dataset.batch(args.batch_size)
    test_dataset = tf.data.TFRecordDataset("cags.test.tfrecord")
    test_dataset = test_dataset.map(CAGS.parse)
    test_dataset = test_dataset.map(lambda example: (example["image"], example["mask"]))
    test_dataset = test_dataset.batch(args.batch_size)

    #backbone.trainable = False
    #model = tf.keras.Sequential()
    #model.add(backbone)
    print(backbone.outputs)
    model = Model(args, CAGS, backbone)

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1)  # maybe delete

    logs = model.fit(
        train_dataset,
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=dev_dataset,
        callbacks=[tb_callback],  # maybe delete
    )






    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the masks on the test set
        #test_masks = model.predict(...)
        test_masks = model.predict(test_dataset)

        for mask in test_masks:
            zeros, ones, runs = 0, 0, []
            for pixel in np.reshape(mask >= 0.5, [-1]):
                if pixel:
                    if zeros or (not zeros and not ones):
                        runs.append(zeros)
                        zeros = 0
                    ones += 1
                else:
                    if ones:
                        runs.append(ones)
                        ones = 0
                    zeros += 1
            runs.append(zeros + ones)
            print(*runs, file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
