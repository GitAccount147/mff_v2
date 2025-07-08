#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from uppercase_data import UppercaseData

# our imports:
import sys
from typing import List, Tuple

# Pepa: 3d76595a-e687-11e9-9ce9-00505601122b
# "no": 594215cf-e687-11e9-9ce9-00505601122b

# TODO: Set reasonable values for the hyperparameters, especially for
# `alphabet_size`, `batch_size`, `epochs`, and `windows`.
# Also, you can set the number of the threads 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=65, type=int, help="If given, use this many most frequent chars.")
# czech language 42 letters; len(set(uppercase_data.test.text)) = 268
parser.add_argument("--batch_size", default=850, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=15, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=8, type=int, help="Window size to use.")

# our hyperparameters:
parser.add_argument("--hidden_layers", default=[1200], nargs="*", type=int, help="Hidden layer sizes.")
# parser.add_argument("--hidden_layer_size", default=512, nargs="*", type=int, help="Hidden layer size.")
parser.add_argument("--dropout", default=0.325, type=float, help="Dropout regularization.")
parser.add_argument("--weight_decay", default=0, type=float, help="Weight decay strength.")
parser.add_argument("--label_smoothing", default=0.5, type=float, help="Label smoothing.")
# parser.add_argument("--labels", default=2, type=int, help="Number of labels.")


def main(args: argparse.Namespace) -> None:
    print(args)

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
    uppercase_data = UppercaseData(args.window, args.alphabet_size)

    #print(uppercase_data.test.text)
    #print(len(set(uppercase_data.test.text)))

    # TODO: Implement a suitable model, optionally including regularization, select
    # good hyperparameters and train the model.
    #
    # The inputs are _windows_ of fixed size (`args.window` characters on left,
    # the character in question, and `args.window` characters on right), where
    # each character is represented by a `tf.int32` index. To suitably represent
    # the characters, you can:
    # - Convert the character indices into _one-hot encoding_. There is no
    #   explicit Keras layer, but you can
    #   - use a Lambda layer which can encompass any function:
    #       tf.keras.Sequential([
    #         tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32),
    #         tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))),
    #         ...
    #       ])
    #   - or use Functional API and then any TF function can be used
    #     as a Keras layer:
    #       inputs = tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32)
    #       encoded = tf.one_hot(inputs, len(uppercase_data.train.alphabet))
    #   You can then flatten the one-hot encoded windows and follow with a dense layer.
    # - Alternatively, you can use `tf.keras.layers.Embedding` (which is an efficient
    #   implementation of one-hot encoding followed by a Dense layer) and flatten afterwards.

    # hyper-parameters:
    labels = 2
    activation = tf.nn.relu
    models_number = 5

    # changes for label smoothing:
    if args.label_smoothing != 0:
        uppercase_data.train.data["labels"] = tf.keras.utils.to_categorical(
            uppercase_data.train.data["labels"], num_classes=labels, dtype='float32')
        uppercase_data.dev.data["labels"] = tf.keras.utils.to_categorical(
            uppercase_data.dev.data["labels"], num_classes=labels, dtype='float32')
        uppercase_data.test.data["labels"] = tf.keras.utils.to_categorical(
            uppercase_data.test.data["labels"], num_classes=labels, dtype='float32')

        uppercase_data.train.data["labels"] = uppercase_data.train.data["labels"] * (1 - args.label_smoothing) + (args.label_smoothing / labels)
        uppercase_data.dev.data["labels"] = uppercase_data.dev.data["labels"] * (1 - args.label_smoothing) + (args.label_smoothing / labels)
        uppercase_data.test.data["labels"] = uppercase_data.test.data["labels"] * (1 - args.label_smoothing) + (args.label_smoothing / labels)

    models = []
    for i in range(models_number):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32))
        model.add(tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))))
        model.add(tf.keras.layers.Flatten(input_shape=[2 * args.window + 1] * args.alphabet_size))
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

        tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1)  # maybe delete

        print("Training model {}: ".format(i + 1), end="", file=sys.stderr, flush=True)
        logs = models[-1].fit(
            uppercase_data.train.data["windows"], uppercase_data.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(uppercase_data.dev.data["windows"], uppercase_data.dev.data["labels"]),
            callbacks=[tb_callback],  # maybe delete
        )
        print("Done", file=sys.stderr)

    d = uppercase_data.test.data["windows"].shape[0]
    d_dev = uppercase_data.dev.data["windows"].shape[0]
    prediction = tf.zeros((d, labels))
    prediction_dev = tf.zeros((d_dev, labels))
    for i in range(models_number):
        prediction += models[i].predict(uppercase_data.test.data["windows"])
        prediction_dev += models[i].predict(uppercase_data.dev.data["windows"])
        #individual_accuracy = models[i].evaluate(uppercase_data.dev.data["windows"], uppercase_data.dev.data["labels"])[
        #    1]
        #print("Accuracy of model", i + 1, ":", individual_accuracy)

    prediction /= models_number
    prediction_dev /= models_number

    if args.label_smoothing != 0:
        metric = tf.metrics.CategoricalAccuracy(name="accuracy")
    else:
        metric = tf.metrics.SparseCategoricalAccuracy(name="accuracy")
    metric.update_state(uppercase_data.dev.data["labels"], prediction_dev)
    ensemble_accuracy = metric.result().numpy()
    print("Ensemble accuracy:", ensemble_accuracy)

    prediction = tf.math.argmax(prediction, axis=1)

    # TODO: Generate correctly capitalized test set.
    # Use `uppercase_data.test.text` as input, capitalize suitable characters,
    # and write the result to predictions_file (which is
    # `uppercase_test.txt` in the `args.logdir` directory).

    final_prediction = ""
    for i in range(len(prediction)):
        if prediction[i] == 1:
            final_prediction += uppercase_data.test.text[i].upper()
        else:
            final_prediction += uppercase_data.test.text[i]

    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "uppercase_test.txt"), "w", encoding="utf-8") as predictions_file:
        predictions_file.write(final_prediction)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
