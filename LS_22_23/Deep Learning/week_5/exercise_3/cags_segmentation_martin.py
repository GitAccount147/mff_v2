#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from cags_dataset import CAGS

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of the threads 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--augment", default=None, choices=["tf_image", "layers"], help="Augmentation type.")
parser.add_argument("--show_images", default=False, action="store_true", help="Show augmented images.")
parser.add_argument("--dropout", default=0.2, help="value of dropout")
parser.add_argument("--weight_decay", default=0.1, help="value of weight_decay")
parser.add_argument("--learning_rate", default=0.001, help="value of learning_rate")
parser.add_argument("--learning_rate_final", default=0.0001, help="value of learning_rate")
parser.add_argument("--label_smoothing", default=False, help="value of learning_rate")


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
    metric= cags.MaskIoUMetric()
    train = cags.train.map(lambda x: (x["image"], x["mask"]))
    dev = cags.dev.map(lambda x: (x["image"], x["mask"]))
    train = train.shuffle(len(train), seed=args.seed)#.map(image_to_float)  #MOZNA PRIDAT BUFFER DO SHUFFLE
    train = train.batch(args.batch_size,drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    dev =dev.batch(args.batch_size,drop_remainder=True).prefetch(tf.data.AUTOTUNE) #dev.map(image_to_float).batch(args.batch_size,drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    test=cags.test.map(lambda x: (x["image"]))

    # Load the EfficientNetV2-B0 model. It assumes the input images are
    # represented in [0-255] range using either `tf.uint8` or `tf.float32` type.

    inputs = tf.keras.Input([CAGS.H, CAGS.W,CAGS.C])
    backbone = tf.keras.applications.EfficientNetV2B0(include_top=False,input_tensor=inputs)
    backbone = tf.keras.Model(
        inputs=backbone.input,
        outputs=[backbone.get_layer(layer).output for layer in [
             "top_activation", "block5e_add", "block3b_add", "block2b_add", "block1a_project_activation"]]
    )

    backbone.trainable = False
    hidden=backbone.output[0]#["top_activation"]

    hidden = tf.keras.layers.Conv2D(112, (3, 3), padding="same", use_bias=False,activation=None)(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Activation('relu')(hidden)

   # hidden = tf.keras.layers.Conv2D(112, (3, 3), padding="same", use_bias=False,activation=None)(hidden)
   # hidden = tf.keras.layers.BatchNormalization()(hidden)
    #hidden = tf.keras.layers.Activation('relu')(hidden)

    hidden = tf.keras.layers.Conv2DTranspose(112, (2, 2),strides=(2,2), padding="same", use_bias=False,activation=None)(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Activation('relu')(hidden)
    hidden = tf.keras.layers.Add()([backbone.output[1],hidden])

    #hidden = tf.keras.layers.Conv2D(48, (3, 3), padding="same", use_bias=False,activation=None)(hidden)
    #hidden = tf.keras.layers.BatchNormalization()(hidden)
    #hidden = tf.keras.layers.Activation('relu')(hidden)

    hidden = tf.keras.layers.Conv2D(48, (3, 3), padding="same", use_bias=False,activation=None)(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Activation('relu')(hidden)

    hidden = tf.keras.layers.Conv2DTranspose(48, (2, 2),strides=(2,2), padding="same", use_bias=False,activation=None)(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Activation('relu')(hidden)
    hidden = tf.keras.layers.Add()([backbone.output[2],hidden])

    #hidden = tf.keras.layers.Conv2D(32, (3, 3), padding="same", use_bias=False,activation=None)(hidden)
    #hidden = tf.keras.layers.BatchNormalization()(hidden)
    #hidden = tf.keras.layers.Activation('relu')(hidden)

    hidden = tf.keras.layers.Conv2D(32, (3, 3), padding="same", use_bias=False,activation=None)(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Activation('relu')(hidden)

    hidden = tf.keras.layers.Conv2DTranspose(32, (2, 2),strides=(2,2), padding="same", use_bias=False,activation=None)(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Activation('relu')(hidden)
    hidden = tf.keras.layers.Add()([backbone.output[3],hidden])

    #hidden = tf.keras.layers.Conv2D(16, (3, 3), padding="same", use_bias=False,activation=None)(hidden)
    #hidden = tf.keras.layers.BatchNormalization()(hidden)
    #hidden = tf.keras.layers.Activation('relu')(hidden)

    hidden = tf.keras.layers.Conv2D(16, (3, 3), padding="same", use_bias=False,activation=None)(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Activation('relu')(hidden)

    hidden = tf.keras.layers.Conv2DTranspose(16, (2, 2),strides=(2,2), padding="same", use_bias=False,activation=None)(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Activation('relu')(hidden)
    hidden = tf.keras.layers.Add()([backbone.output[4],hidden])

    #hidden = tf.keras.layers.Conv2D(16, (3, 3), padding="same", use_bias=False,activation=None)(hidden)
    #hidden = tf.keras.layers.BatchNormalization()(hidden)
    #hidden = tf.keras.layers.Activation('relu')(hidden)

    hidden = tf.keras.layers.Conv2D(16, (3, 3), padding="same", use_bias=False,activation=None)(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Activation('relu')(hidden)

    hidden = tf.keras.layers.Conv2DTranspose(16, (2, 2),strides=(2,2), padding="same", use_bias=False,activation=None)(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Activation('relu')(hidden)

    #hidden = tf.keras.layers.Conv2D(8, (3, 3), padding="same", use_bias=False,activation=None)(hidden)
    #hidden = tf.keras.layers.BatchNormalization()(hidden)
    #hidden = tf.keras.layers.Activation('relu')(hidden)

    #hidden2 = tf.keras.layers.Conv2D(8, (3, 3), padding="same", use_bias=False,activation=None)(inputs)
    #hidden2 = tf.keras.layers.BatchNormalization()(hidden2)
    #hidden2 = tf.keras.layers.Activation('relu')(hidden2)

    #hidden2 = tf.keras.layers.Conv2D(8, (3, 3), padding="same", use_bias=False,activation=None)(inputs)
    #hidden2 = tf.keras.layers.BatchNormalization()(hidden2)
    #hidden2 = tf.keras.layers.Activation('relu')(hidden2)

    #hidden = tf.keras.layers.Conv2D(8, (3, 3), padding="same", use_bias=False,activation=None)(hidden)
    #hidden = tf.keras.layers.BatchNormalization()(hidden)
    #hidden = tf.keras.layers.Activation('relu')(hidden)
    #
    #hidden = tf.keras.layers.Conv2D(8, (3, 3), padding="same", use_bias=False,activation=None)(hidden)
    #hidden = tf.keras.layers.BatchNormalization()(hidden)
    #hidden = tf.keras.layers.Activation('relu')(hidden)

    #hidden = tf.keras.layers.Concatenate()([hidden2,hidden])

    #hidden = tf.keras.layers.Conv2D(8, (3, 3), padding="same", use_bias=False,activation=None)(hidden)
    #hidden = tf.keras.layers.BatchNormalization()(hidden)
    #hidden = tf.keras.layers.Activation('relu')(hidden)


    hidden = tf.keras.layers.Conv2D(8, (3, 3), padding="same", use_bias=False,activation=None)(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Activation('relu')(hidden)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(hidden)


    model = tf.keras.Model(inputs, outputs)


    model.compile(
        optimizer=tf.optimizers.Adam(jit_compile=False,weight_decay=args.weight_decay,learning_rate=args.learning_rate),#,learning_rate=args.learning_rate,weight_decay=args.weight_decay),
        loss=tf.keras.losses.BinaryFocalCrossentropy(),
        metrics=[metric],
    )



    model.fit(train, epochs=args.epochs, validation_data=dev)


    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the masks on the test set
        test_masks = model.predict(test.map(lambda x: tf.expand_dims(x, 0) ))

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