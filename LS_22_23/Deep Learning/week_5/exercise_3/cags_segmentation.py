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

# newly added
from typing import Dict, Tuple
#import tensorflow_datasets as tfds
#from tensorflow_examples.models.pix2pix import pix2pix

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of the threads 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=5, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

# our hyperparameters:
#parser.add_argument("--cnn", default="CB-8-3-1-same,CB-8-3-1-same,M-3-2,CB-16-3-1-same,CB-16-3-1-same,M-3-2,F,H-100", type=str, help="CNN architecture.")

class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace, dataset, back) -> None:
        #inputs = tf.keras.layers.Input(shape=[dataset.H, dataset.W, dataset.C])
        inputs = back.inputs

        top_activation, block5e_add, block3b_add, block2b_add, block1a_project_activation = back(inputs)
        hidden = block1a_project_activation
        #hidden = block2b_add

        tr_conv_1 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=2, strides=2, padding="same")
        conv_1 = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding="same")
        bn = tf.keras.layers.BatchNormalization()
        relu = tf.nn.relu

        #hidden = relu(bn(conv_1(tr_conv_1(hidden))))
        #hidden = relu(bn(tr_conv_1(hidden)))
        #hidden = relu(bn(hidden))
        #hidden = relu(tr_conv_1(hidden))
        hidden = tr_conv_1(hidden)

        conv_2 = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, padding="same")
        den = tf.keras.layers.Dense(256, activation=tf.nn.softmax)
        sig = tf.nn.sigmoid
        soft = tf.nn.softmax

        #hidden = sig(den(conv_2(hidden)))
        #hidden = sig(den(hidden))
        hidden = soft(conv_2(tr_conv_1(hidden)))

        outputs = hidden


        #hidden = tf.keras.layers.Dense(256)(block1a_project_activation)
        #hidden = tf.keras.layers.Flatten()(hidden)
        #hidden = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=2, strides=2)(block1a_project_activation)
        #hidden = tf.nn.relu(hidden)

        # Add the final output layer
        #outputs = tf.keras.layers.Dense([dataset.H, dataset.W, dataset.C], activation=tf.nn.softmax)(hidden)
        #hidden = tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=2, strides=2)(hidden)
        #hidden = tf.keras.layers.BatchNormalization()(hidden)
        #hidden = tf.nn.relu(hidden)

        #hidden = tf.keras.layers.Conv2D(filters=256, kernel_size=1, strides=1)(hidden)

        #hidden = tf.keras.layers.Dense(1, activation=tf.nn.softmax)(hidden)
        #outputs = tf.nn.sigmoid(hidden)

        super().__init__(inputs=inputs, outputs=outputs)
        self.compile(
            optimizer=tf.optimizers.Adam(jit_compile=False),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            #loss=tf.losses.BinaryCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
            #metrics=[tf.metrics.BinaryAccuracy(name="accuracy")],
        )
        #print(back.trainable_variables)
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

"""
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels:int):
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])

  # Downsampling through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)
"""

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
    #backbone = tf.keras.applications.EfficientNetV2B0(include_top=False, input_shape=[224, 224, 3])
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

    #for example in cags.train:
    #    print(example)  #  ... 'mask': <tf.Tensor: shape=(224, 224, 1), dtype=uint8, numpy=...> ...

    #print(backbone.outputs)
    #print(backbone.inputs)

    #def image_to_float(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    #    return tf.image.convert_image_dtype(image, tf.float32), label

    def image_to_float(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return image, tf.cast(label, tf.uint8)

    train_dataset = tf.data.TFRecordDataset("cags.train.tfrecord")
    train_dataset = train_dataset.map(CAGS.parse)
    train_dataset = train_dataset.map(lambda example: (example["image"], example["mask"]))
    #train_dataset = train_dataset.map(image_to_float)
    train_dataset = train_dataset.batch(args.batch_size)

    dev_dataset = tf.data.TFRecordDataset("cags.dev.tfrecord")
    dev_dataset = dev_dataset.map(CAGS.parse)
    dev_dataset = dev_dataset.map(lambda example: (example["image"], example["mask"]))
    #dev_dataset = dev_dataset.map(image_to_float)
    dev_dataset = dev_dataset.batch(args.batch_size)

    test_dataset = tf.data.TFRecordDataset("cags.test.tfrecord")
    test_dataset = test_dataset.map(CAGS.parse)
    test_dataset = test_dataset.map(lambda example: (example["image"], example["mask"]))
    #test_dataset = test_dataset.map(image_to_float)
    test_dataset = test_dataset.batch(args.batch_size)


    #"""
    #print(backbone.trainable_variables)
    backbone.trainable = False
    #print(backbone.trainable_variables)


    #model = tf.keras.Sequential()
    #model.add(backbone)
    #print(backbone.outputs)
    model = Model(args, CAGS, backbone)
    #"""


    """
    # Sequential:
    backbone2 = tf.keras.applications.EfficientNetV2B0(include_top=False, pooling="avg")
    model = tf.keras.Sequential()
    backbone2.trainable = False
    model.add(backbone.get_layer("block1a_project_activation"))
    #model.add(backbone2)
    #model.add(tf.keras.layers.Dense(100))
    #model.add(tf.keras.layers.Dense(224*224, activation=tf.nn.softmax))
    model.add(tf.keras.layers.Conv2DTranspose(filters=1280, kernel_size=2, strides=5))
    #model.add(tf.keras.layers.Flatten())

    optimizer = tf.optimizers.experimental.AdamW()
    optimizer.exclude_from_weight_decay(var_names=['bias'])

    model.compile(
        optimizer=optimizer,
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    """

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
