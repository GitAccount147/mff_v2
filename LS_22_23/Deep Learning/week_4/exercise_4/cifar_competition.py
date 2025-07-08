#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from cifar10 import CIFAR10

# our imports:
import sys
from typing import Dict, Tuple

# Supr Dupr Tym:
# 3d76595a-e687-11e9-9ce9-00505601122b
# 594215cf-e687-11e9-9ce9-00505601122b

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

# our hyperparameters:
#parser.add_argument("--hidden_layers", default=[100], nargs="*", type=int, help="Hidden layer sizes.")
#parser.add_argument("--dropout", default=0, type=float, help="Dropout regularization.")
#parser.add_argument("--weight_decay", default=0, type=float, help="Weight decay strength.")
parser.add_argument("--label_smoothing", default=0, type=float, help="Label smoothing.")
parser.add_argument("--augment", default=None, choices=["tf_image", "layers"], help="Augmentation type.")


class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace, cnn_arch: str) -> None:
        inputs = tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])

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
        outputs = tf.keras.layers.Dense(len(CIFAR10.LABELS), activation=tf.nn.softmax)(hidden)

        super().__init__(inputs=inputs, outputs=outputs)
        self.compile(
            optimizer=tf.optimizers.Adam(jit_compile=False),
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
        tf.data.experimental.enable_debug_mode()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load data
    cifar = CIFAR10()

    # TODO: Create the model and train it
    #model = ...

    # hyper-parameters:
    labels = len(CIFAR10.LABELS)
    #activation = tf.nn.relu
    models_number = 1
    cnn_types = ["CB-8-3-5-valid,R-[CB-8-3-1-same,CB-8-3-1-same],F,H-50",
                 "CB-8-3-5-valid,R-[CB-8-3-1-same,CB-8-3-1-same],F,H-100",
                 "CB-8-3-5-valid,R-[CB-8-3-1-same,CB-8-3-1-same],F,H-500",
                 "CB-8-3-1-same,CB-8-3-1-same,M-3-2,CB-16-3-1-same,CB-16-3-1-same,M-3-2,F,H-100",
                 "CB-8-3-1-same,CB-8-3-1-same,M-3-2,CB-16-3-1-same,CB-16-3-1-same,M-3-2,F,H-500",
                 "CB-8-3-1-same,CB-8-3-1-same,M-3-2,CB-16-3-1-same,CB-16-3-1-same,M-3-2,F,H-500,D-0.5",
                 "CB-8-3-1-same,CB-8-3-1-same,M-3-2,CB-16-3-1-same,CB-16-3-1-same,M-3-2,F,H-1000,D-0.5",
                 "CB-8-3-5-valid,R-[CB-8-3-1-same,CB-8-3-1-same],F,H-500,D-0.5",
                 "C-32-3-1-same,C-32-3-1-same,M-2-2,D-0.25,C-64-3-1-same,C-64-3-1-same,M-2-2,D-0.25,F,H-512,D-0.5",  # https://www.kaggle.com/code/roblexnana/cifar10-with-cnn-for-beginer
                 "F,H-500,D-0.5",
                 "CB-32-3-1-same,CB-32-3-1-same,M-2-2,D-0.25,CB-64-3-1-same,CB-64-3-1-same,M-2-2,D-0.25,F,H-512,D-0.5",
                 "CB-32-3-1-same,CB-32-3-1-same,M-2-2,D-0.2,CB-64-3-1-same,CB-64-3-1-same,M-2-2,D-0.3,CB-128-3-1-same,CB-128-3-1-same,M-2-2,D-0.4,F,H-128,D-0.5",  # https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
                 "CB-32-3-1-same,CB-32-3-1-same,M-2-2,D-0.1,CB-64-3-1-same,CB-64-3-1-same,M-2-2,D-0.2,CB-128-3-1-same,CB-128-3-1-same,M-2-2,D-0.3,CB-256-3-1-same,CB-256-3-1-same,M-2-2,D-0.4,F,H-64,D-0.5",
                 "CB-32-3-1-same,CB-32-3-1-same,M-2-2,D-0.2,CB-64-3-1-same,CB-64-3-1-same,M-2-2,D-0.3,CB-128-3-1-same,CB-128-3-1-same,M-2-2,D-0.4,F,H-512,D-0.5",
                 #"R-[CB-32-3-1-same,CB-32-3-1-same],M-2-2,D-0.25,CB-64-3-1-same,CB-64-3-1-same,M-2-2,D-0.25,F,H-512,D-0.5"
                 ]
    cnn_arch = cnn_types[-1]

    print(args)
    print(cnn_arch)

    #print(cifar.test.data["labels"])

    # changes for label smoothing:
    if args.label_smoothing != 0:
        cifar.train.data["labels"] = tf.keras.utils.to_categorical(
            cifar.train.data["labels"], num_classes=labels, dtype='float32')
        cifar.dev.data["labels"] = tf.keras.utils.to_categorical(
            cifar.dev.data["labels"], num_classes=labels, dtype='float32')
        cifar.test.data["labels"] = tf.keras.utils.to_categorical(
            cifar.test.data["labels"], num_classes=labels, dtype='float32')

        cifar.train.data["labels"] = cifar.train.data["labels"] * (1 - args.label_smoothing) + (
                    args.label_smoothing / labels)
        cifar.dev.data["labels"] = cifar.dev.data["labels"] * (1 - args.label_smoothing) + (
                    args.label_smoothing / labels)
        cifar.test.data["labels"] = cifar.test.data["labels"] * (1 - args.label_smoothing) + (
                    args.label_smoothing / labels)

    #print(cifar.train.data["images"].shape)

    train = tf.data.Dataset.from_tensor_slices((cifar.train.data["images"], cifar.train.data["labels"]))
    dev = tf.data.Dataset.from_tensor_slices((cifar.dev.data["images"], cifar.dev.data["labels"]))

    # Convert images from tf.uint8 to tf.float32 and scale them to [0, 1] in the process.
    def image_to_float(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return tf.image.convert_image_dtype(image, tf.float32), label

    # Simple data augmentation using `tf.image`.
    generator = tf.random.Generator.from_seed(args.seed)

    def train_augment_tf_image(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if generator.uniform([]) >= 0.5:
            image = tf.image.flip_left_right(image)
        image = tf.image.resize_with_crop_or_pad(image, CIFAR10.H + 6, CIFAR10.W + 6)
        image = tf.image.resize(image, [generator.uniform([], CIFAR10.H, CIFAR10.H + 12 + 1, dtype=tf.int32),
                                        generator.uniform([], CIFAR10.W, CIFAR10.W + 12 + 1, dtype=tf.int32)])
        image = tf.image.crop_to_bounding_box(
            image, target_height=CIFAR10.H, target_width=CIFAR10.W,
            offset_height=generator.uniform([], maxval=tf.shape(image)[0] - CIFAR10.H + 1, dtype=tf.int32),
            offset_width=generator.uniform([], maxval=tf.shape(image)[1] - CIFAR10.W + 1, dtype=tf.int32),
        )
        return image, label

    # Simple data augmentation using layers.
    def train_augment_layers(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.keras.layers.RandomFlip("horizontal", seed=args.seed)(image)  # Bug, flip always; fixed in TF 2.12.
        image = tf.keras.layers.RandomZoom(0.2, seed=args.seed)(image)
        image = tf.keras.layers.RandomTranslation(0.15, 0.15, seed=args.seed)(image)
        image = tf.keras.layers.RandomRotation(0.1, seed=args.seed)(image)  # Does not always help (too blurry?).
        return image, label


    number_to_aug = 45000
    train = train.take(number_to_aug)
    train = train.shuffle(number_to_aug, seed=args.seed)
    train = train.map(image_to_float)
    if args.augment == "tf_image":
        train = train.map(train_augment_tf_image)
    if args.augment == "layers":
        train = train.map(train_augment_layers)
    train = train.batch(args.batch_size)
    train = train.prefetch(tf.data.AUTOTUNE)  # optional

    dev = dev.map(image_to_float)
    dev = dev.batch(args.batch_size)
    dev = dev.prefetch(tf.data.AUTOTUNE)  # optional


    def img_to_float_2(image: tf.Tensor) -> tf.Tensor:
        return tf.image.convert_image_dtype(image, tf.float32)
    #test = tf.data.Dataset.from_tensor_slices(cifar.test.data["images"])
    #test = test.map(image_to_float_test)
    #test = test.batch(args.batch_size)
    #test = test.prefetch(tf.data.AUTOTUNE)  # optional


    models = []
    for i in range(models_number):
        # old sequential model:
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C]))  # , dtype=tf.int32
        model.add(tf.keras.layers.Flatten())
        for j in range(len(args.hidden_layers)):
            model.add(tf.keras.layers.Dense(args.hidden_layers[j], activation=activation))
            model.add(tf.keras.layers.Dropout(rate=args.dropout))
        model.add(tf.keras.layers.Dense(labels, activation=tf.nn.softmax))

        models.append(model)

        optimizer = tf.optimizers.experimental.AdamW(weight_decay=args.weight_decay)
        optimizer.exclude_from_weight_decay(var_names=['bias'])

        if args.label_smoothing != 0:
            loss = tf.losses.CategoricalCrossentropy()
            metrics = [tf.metrics.CategoricalAccuracy(name="accuracy")]
        else:
            loss = tf.losses.SparseCategoricalCrossentropy()
            metrics = [tf.metrics.SparseCategoricalAccuracy(name="accuracy")]

        models[-1].compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )
        """

        model = Model(args, cnn_arch)
        models.append(model)

        tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1)  # maybe delete

        print("Training model {}: ".format(i + 1), end="", file=sys.stderr, flush=True)
        logs = models[-1].fit(
            train,
            #cifar.train.data["images"], cifar.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            #validation_data=(cifar.dev.data["images"], cifar.dev.data["labels"]),
            validation_data=dev,
            callbacks=[tb_callback],  # maybe delete
        )
        print("Done", file=sys.stderr)

    d = cifar.test.data["images"].shape[0]
    d_dev = cifar.dev.data["images"].shape[0]
    prediction = tf.zeros((d, labels))
    prediction_dev = tf.zeros((d_dev, labels))
    #dev_im = tf.data.Dataset.from_tensor_slices((cifar.dev.data["images"]))
    #dev_im = tf.data.Dataset(cifar.dev.data["images"]).map(img_to_float_2)
    dev_2 = tf.data.Dataset.from_tensor_slices((cifar.dev.data["images"], cifar.dev.data["labels"]))
    dev_2 = dev_2.map(image_to_float)
    test_2 = tf.data.Dataset.from_tensor_slices((cifar.test.data["images"], cifar.test.data["labels"]))
    test_2 = test_2.map(image_to_float)


    def get_img(image: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        return tf.image.convert_image_dtype(image, tf.float32)
    dev_2 = dev_2.map(get_img)
    dev_2 = dev_2.batch(args.batch_size)
    test_2 = test_2.map(get_img)
    test_2 = test_2.batch(args.batch_size)

    for i in range(models_number):
        #prediction += models[i].predict()
        #prediction += models[i].predict(cifar.test.data["images"], batch_size=args.batch_size)
        prediction += models[i].predict(test_2)
        #prediction += models[i].predict(test)
        dev_pred_ind = models[i].predict(dev_2)
        #dev_pred_ind = models[i].predict(cifar.dev.data["images"], batch_size=args.batch_size)
        #dev_pred_ind = models[i].predict(dev_im)
        prediction_dev += dev_pred_ind
        #individual_accuracy = models[i].evaluate(cifar.dev.data["images"], cifar.dev.data["labels"])[1]
        individual_accuracy = models[i].evaluate(dev)[1]
        #some_metric = tf.metrics.CategoricalAccuracy(name="accuracy")
        #some_metric.update_state(cifar.dev.data["labels"], dev_pred_ind)
        #print("Measure 1:", some_metric.result().numpy())
        #print("Measure 2:", models[i].evaluate(dev_2)[1])
        print("Accuracy of model", i + 1, ":", individual_accuracy)

    prediction /= models_number
    prediction_dev /= models_number

    if args.label_smoothing != 0:
        metric = tf.metrics.CategoricalAccuracy(name="accuracy")
    else:
        metric = tf.metrics.SparseCategoricalAccuracy(name="accuracy")
    metric.update_state(cifar.dev.data["labels"], prediction_dev)
    #print(cifar.dev.data["labels"].shape, tf.math.argmax(prediction_dev, axis=1).shape, prediction_dev.shape)
    #metric.update_state(cifar.dev.data["labels"], tf.math.argmax(prediction_dev, axis=1))
    #metric.update_state(dev_im, prediction_dev)
    ensemble_accuracy = metric.result().numpy()
    print("Ensemble accuracy:", ensemble_accuracy)

    #prediction = tf.math.argmax(prediction, axis=1)









    # pre-made code:
    """
    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as predictions_file:
        for probs in model.predict(cifar.test.data["images"], batch_size=args.batch_size):
            print(np.argmax(probs), file=predictions_file)
    """

    # our code:
    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as predictions_file:
        for probs in prediction:
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
