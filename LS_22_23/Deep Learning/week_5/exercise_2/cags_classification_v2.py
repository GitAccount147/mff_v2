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
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

# our hyperparameters:
parser.add_argument("--hidden_layers", default=[100], nargs="*", type=int, help="Hidden layer sizes.")
parser.add_argument("--dropout", default=0.3, type=float, help="Dropout regularization.")
parser.add_argument("--learning_rate", default=0.00001, type=float, help="Fine-tuning learning rate.")
parser.add_argument("--epochs_ft", default=1, type=int, help="Number of epochs for Fine-tuning.")


class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace, cnn_arch: str) -> None:
        inputs = tf.keras.layers.Input(shape=[CAGS.H, CAGS.W, CAGS.C])

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

        super().__init__(inputs=inputs, outputs=outputs)
        self.compile(
            optimizer=tf.optimizers.Adam(jit_compile=False),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

def main(args: argparse.Namespace) -> None:
    print("ARGS:", args)
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

    # Load the data. Note that both the "image" and the "mask" images
    # are represented using `tf.uint8`s in [0-255] range.
    cags = CAGS()

    #for example in cags.train:
    #    print(example)

    train_dataset = tf.data.TFRecordDataset("cags.train.tfrecord")
    train_dataset = train_dataset.map(CAGS.parse)
    train_dataset = train_dataset.map(lambda example: (example["image"], example["label"]))
    train_dataset = train_dataset.batch(args.batch_size)
    dev_dataset = tf.data.TFRecordDataset("cags.dev.tfrecord")
    dev_dataset = dev_dataset.map(CAGS.parse)
    dev_dataset = dev_dataset.map(lambda example: (example["image"], example["label"]))
    dev_dataset = dev_dataset.batch(args.batch_size)
    test_dataset = tf.data.TFRecordDataset("cags.test.tfrecord")
    test_dataset = test_dataset.map(CAGS.parse)
    test_dataset = test_dataset.map(lambda example: (example["image"], example["label"]))
    test_dataset = test_dataset.batch(args.batch_size)

    # Load the EfficientNetV2-B0 model. It assumes the input images are
    # represented in [0-255] range using either `tf.uint8` or `tf.float32` type.
    backbone = tf.keras.applications.EfficientNetV2B0(include_top=False, pooling="avg")

    # TODO: Create the model and train it
    #model = ...


    backbone.trainable = False
    #print(backbone.layers)
    model = tf.keras.Sequential()
    model.add(backbone)
    #model.add(tf.keras.layers.Input(shape=[cags.H, cags.W, cags.C], dtype=tf.uint8))
    #model.add(tf.keras.layers.Flatten())
    for i in range(len(args.hidden_layers)):
        model.add(tf.keras.layers.Dense(args.hidden_layers[i]))
        model.add(tf.keras.layers.Dropout(rate=args.dropout))
    model.add(tf.keras.layers.Dense(len(cags.LABELS), activation=tf.nn.softmax))

    optimizer = tf.optimizers.experimental.AdamW()
    optimizer.exclude_from_weight_decay(var_names=['bias'])

    model.compile(
        optimizer=optimizer,
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1)  # maybe delete

    logs = model.fit(
        #cags.train["image"], cags.train["label"],
        train_dataset,
        batch_size=args.batch_size, epochs=args.epochs,
        #validation_data=(cags.dev["image"], cags.dev["label"]),
        validation_data=dev_dataset,
        callbacks=[tb_callback],  # maybe delete
    )


    # FineTuning:
    backbone.trainable = True
    optimizer2 = tf.optimizers.experimental.AdamW(learning_rate=args.learning_rate)
    optimizer2.exclude_from_weight_decay(var_names=['bias'])
    #optimizer.build(model.trainable_variables)
    model.compile(
        optimizer=optimizer2,
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    logs = model.fit(
        # cags.train["image"], cags.train["label"],
        train_dataset,
        batch_size=args.batch_size, epochs=args.epochs_ft,
        # validation_data=(cags.dev["image"], cags.dev["label"]),
        validation_data=dev_dataset,
        callbacks=[tb_callback],  # maybe delete
    )




    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_classification.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the probabilities on the test set
        #test_probabilities = model.predict(...)
        test_probabilities = model.predict(test_dataset)

        for probs in test_probabilities:
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
