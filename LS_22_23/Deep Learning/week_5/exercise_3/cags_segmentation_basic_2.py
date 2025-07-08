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
parser.add_argument("--batch_size", default=5, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


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

    train_dataset = tf.data.TFRecordDataset("cags.train.tfrecord")
    train_dataset = train_dataset.map(CAGS.parse)
    train_dataset = train_dataset.map(lambda example: (example["image"], example["mask"]))
    # train_dataset = train_dataset.map(image_to_float)
    train_dataset = train_dataset.batch(args.batch_size)

    dev_dataset = tf.data.TFRecordDataset("cags.dev.tfrecord")
    dev_dataset = dev_dataset.map(CAGS.parse)
    dev_dataset = dev_dataset.map(lambda example: (example["image"], example["mask"]))
    # dev_dataset = dev_dataset.map(image_to_float)
    dev_dataset = dev_dataset.batch(args.batch_size)

    test_dataset = tf.data.TFRecordDataset("cags.test.tfrecord")
    test_dataset = test_dataset.map(CAGS.parse)
    test_dataset = test_dataset.map(lambda example: (example["image"], example["mask"]))
    # test_dataset = test_dataset.map(image_to_float)
    test_dataset = test_dataset.batch(args.batch_size)

    test_dataset2 = tf.data.TFRecordDataset("cags.test.tfrecord")
    test_dataset2 = test_dataset2.map(CAGS.parse)
    test_dataset2 = test_dataset2.map(lambda example: (example["mask"]))
    test_dataset2 = test_dataset2.batch(1)

    #print(test_dataset2)



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

    #print(backbone.outputs)
    #print(backbone.inputs)
    #print(backbone.inputs[0])
    #print(backbone.layers)
    #a = backbone.predict(test_dataset2.take(1))
    #print(np.array(a).shape)
    #print(np.array(a[0][0]).shape)  #(7, 7, 1280)
    #print(np.array(a[1][0]).shape)  #(14, 14, 112)
    #print(np.array(a[2][0]).shape)  #(28, 28, 48)
    #print(np.array(a[3][0]).shape)  #(56, 56, 32)
    #print(np.array(a[4][0]).shape)  #(112, 112, 16)


    # TODO: Create the model and train it
    # model = backbone

    '''
    # layer 1
    tconv_L1 = tf.keras.layers.Conv2DTranspose(kernel_size=2, strides=2, filters=1280)(backbone.outputs[0])
    batch_norm_L1 = tf.keras.layers.BatchNormalization()(tconv_L1)
    relu_L1 = tf.keras.activations.relu(batch_norm_L1)


    # layer 2
    conv_L2_1 = tf.keras.layers.Conv2D(kernel_size=3, filters=112)(relu_L1+backbone.outputs[1])
    batch_norm_L2_1 = tf.keras.layers.BatchNormalization()(conv_L2_1)
    relu_L2_1 = tf.keras.activations.relu(batch_norm_L2_1)
    conv_L2_2 = tf.keras.layers.Conv2D(kernel_size=3, filters=112)(relu_L2_1)
    batch_norm_L2_2 = tf.keras.layers.BatchNormalization()(conv_L2_2)
    relu_L2_2 = tf.keras.activations.relu(batch_norm_L2_2)

    tconv_L2 = tf.keras.layers.Conv2DTranspose(kernel_size=2, strides=2, filters=112)(relu_L2_2)
    batch_norm_L2_3 = tf.keras.layers.BatchNormalization()(tconv_L2)
    relu_L2_3 = tf.keras.activations.relu(batch_norm_L2_3)


    # layer 3
    conv_L3_1 = tf.keras.layers.Conv2D(kernel_size=3, filters=48)(relu_L2_3 + backbone.outputs[2])
    batch_norm_L3_1 = tf.keras.layers.BatchNormalization()(conv_L3_1)
    relu_L3_1 = tf.keras.activations.relu(batch_norm_L3_1)
    conv_L3_2 = tf.keras.layers.Conv2D(kernel_size=3, filters=48)(relu_L3_1)
    batch_norm_L3_2 = tf.keras.layers.BatchNormalization()(conv_L3_2)
    relu_L3_2 = tf.keras.activations.relu(batch_norm_L3_2)

    tconv_L3 = tf.keras.layers.Conv2DTranspose(kernel_size=2, strides=2, filters=48)(relu_L3_2)
    batch_norm_L3_3 = tf.keras.layers.BatchNormalization()(tconv_L3)
    relu_L3_3 = tf.keras.activations.relu(batch_norm_L3_3)


    # layer 4
    conv_L3_1 = tf.keras.layers.Conv2D(kernel_size=3, filters=32)(relu_L2_3 + backbone.outputs[3])
    batch_norm_L3_1 = tf.keras.layers.BatchNormalization()(conv_L3_1)
    relu_L3_1 = tf.keras.activations.relu(batch_norm_L3_1)
    conv_L3_2 = tf.keras.layers.Conv2D(kernel_size=3, filters=48)(relu_L3_1)
    batch_norm_L3_2 = tf.keras.layers.BatchNormalization()(conv_L3_2)
    relu_L3_2 = tf.keras.activations.relu(batch_norm_L3_2)

    tconv_L3 = tf.keras.layers.Conv2DTranspose(kernel_size=2, strides=2, filters=48)(relu_L3_2)
    batch_norm_L3_3 = tf.keras.layers.BatchNormalization()(tconv_L3)
    relu_L3_3 = tf.keras.activations.relu(batch_norm_L3_3)'''

    backbone.trainable = False

    '''layers = backbone.outputs
    layers.append(backbone.inputs[0])
    #print(layers)

    conv1 = tf.keras.layers.Conv2D(kernel_size=3, filters=backbone.outputs[0].shape[3], padding='same', input_shape=(7,7,1280))(backbone.outputs[0])
    batch_norm1 = tf.keras.layers.BatchNormalization()(conv1)
    relu1 = tf.keras.activations.relu(batch_norm1)
    conv2 = tf.keras.layers.Conv2D(kernel_size=3, filters=backbone.outputs[0].shape[3], padding='same', input_shape=(7,7,1280))(relu1)
    batch_norm2 = tf.keras.layers.BatchNormalization()(conv2)
    relu2 = tf.keras.activations.relu(batch_norm2)


    for i in range(5):
        # up
        tconv = tf.keras.layers.Conv2DTranspose(kernel_size=2, strides=2, filters=layers[i].shape[3], padding='same')(relu2)
        batch_norm3 = tf.keras.layers.BatchNormalization()(tconv)
        relu3 = tf.keras.activations.relu(batch_norm3)

        #concat = tf.keras.layers.Concatenate()([relu3, layers[i+1]])

        # right
        conv3 = tf.keras.layers.Conv2D(kernel_size=3, filters=layers[i+1].shape[3], padding='same')(relu3)
        batch_norm4 = tf.keras.layers.BatchNormalization()(conv3)
        relu4 = tf.keras.activations.relu(batch_norm4)
        conv4 = tf.keras.layers.Conv2D(kernel_size=3, filters=layers[i+1].shape[3], padding='same')(relu4)
        batch_norm5 = tf.keras.layers.BatchNormalization()(conv4)
        relu2 = tf.keras.activations.relu(batch_norm5)

    #flat_conv = tf.keras.layers.Conv2D(kernel_size=1, filters=1)(relu2)
    #print(flat_conv)
    flat_conv = tf.keras.layers.Conv2D(kernel_size=1, filters=1, input_shape=(224, 224, 3))(relu2)
    flatten = tf.keras.layers.Flatten(input_shape=(224, 224, 3))(flat_conv)
    print(flatten.shape)
    out = tf.keras.layers.Dense(224*224, activation=tf.nn.sigmoid, input_shape=(224*224,1))(flatten)'''
    out = tf.keras.layers.Conv2DTranspose(kernel_size=2, strides=2, filters=112, padding='same', input_shape=(7,7,1280))(backbone.outputs[0])
    out = tf.keras.layers.Conv2DTranspose(kernel_size=2, strides=2, filters=48, padding='same', input_shape=(14,14,112))(out)
    out = tf.keras.layers.Conv2DTranspose(kernel_size=2, strides=2, filters=32, padding='same', input_shape=(28,28,48))(out)
    out = tf.keras.layers.Conv2DTranspose(kernel_size=2, strides=2, filters=16, padding='same', input_shape=(56,56,32))(out)
    out = tf.keras.layers.Conv2DTranspose(kernel_size=2, strides=2, filters=3, padding='same', input_shape=(112,112,16))(out)
    #print(out)
    out = tf.keras.layers.Conv2D(kernel_size=1, filters=1, input_shape=(224, 224, 3))(out)
    out = tf.nn.sigmoid(out)
    '''flatten = tf.keras.layers.Flatten(input_shape=(224, 224, 1))(out)'''
    #out = tf.keras.layers.Dense(256, activation=tf.nn.sigmoid, input_shape=(224 * 224, 1))(out)

    model = tf.keras.Model(inputs=backbone.inputs, outputs=out)

    optimizer = tf.optimizers.experimental.AdamW()
    optimizer.exclude_from_weight_decay(var_names=['bias'])

    model.compile(
        optimizer=optimizer,
        loss=tf.losses.CategoricalCrossentropy(),
        metrics=[tf.metrics.CategoricalAccuracy(name="accuracy")],
    )

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
        test_masks = model.predict(test_dataset)
        #print(test_masks.shape)

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
