#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from homr_dataset import HOMRDataset

# "Vtipny komentar o techto osobach"
# 3d76595a-e687-11e9-9ce9-00505601122b
# 594215cf-e687-11e9-9ce9-00505601122b

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=2, type=int, help="Batch size.")
parser.add_argument("--debug", default=True, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

# our:
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay.")


class MargeSimpson(tf.keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        shape = [None, None, HOMRDataset.C]
        #shape = [None, None, None, HOMRDataset.C]
        #shape = [None, HOMRDataset.C]
        #shape = [args.batch_size, None, None, HOMRDataset.C]
        #inputs = tf.keras.layers.Input(shape=shape)
        inputs = tf.keras.layers.Input(shape=shape, ragged=True)
        print("inputs.shape:", inputs.shape)  # [None, None, None, 1] ~ [BatchSize, Height, Width, Channels]

        #inputs = tf.image.resize(inputs, [224, 224])

        widths = inputs.row_lengths(axis=2)
        heights = inputs.row_lengths(axis=1)
        hidden = inputs.to_tensor()
        print("hidden.shape", hidden.shape)
        #hidden = inputs.values

        stride = 1
        conv1 = tf.keras.layers.Conv2D(filters=1, kernel_size=5, strides=(1, stride), padding="same", use_bias=False)
        #conv2 = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=(1, 3), padding="valid", use_bias=False)
        rnn = tf.keras.layers.LSTM(units=32, return_sequences=True)
        bid = tf.keras.layers.Bidirectional(layer=rnn, merge_mode='sum')

        hidden = conv1(hidden)
        hidden = tf.squeeze(hidden, [-1])

        # recalculate widths:
        new_widths = tf.cast(tf.math.ceil(widths / stride), dtype=tf.int64)
        #new_widths = widths

        # prepare for transposing:
        widths_short = new_widths.to_tensor()[:, 0]

        list_comp = [tf.convert_to_tensor([heights[i] for _ in range(widths_short[i])]) for i in
                     range(tf.shape(widths_short)[0])]
        heights_long = tf.ragged.stack(list_comp)
        perm = [0, 1, 3, 2]

        trans = tf.transpose(hidden, perm=perm)
        hidden = tf.RaggedTensor.from_tensor(trans, lengths=[widths_short, heights_long.values])
        # end of transposing


        #hidden = tf.RaggedTensor.from_tensor(hidden, lengths=[heights, new_widths.values])
        #hidden = tf.squeeze(hidden, [-1])
        #print("hidden.shape", hidden.shape)
        #hidden = tf.reshape(hidden, hidden.shape[:-1])

        #hidden = hidden.values
        #hidden = tf.expand_dims(hidden, axis=-1)
        hidden = bid(hidden)

        # Add the final output layer
        units = 1 + len(HOMRDataset.MARKS)
        outputs = tf.keras.layers.Dense(units, activation=tf.nn.softmax)(hidden)

        super().__init__(inputs=inputs, outputs=outputs)

        self.compile(
            optimizer=tf.optimizers.Adam(jit_compile=False, learning_rate=args.learning_rate,
                                         weight_decay=args.weight_decay),
            #loss=tf.losses.SparseCategoricalCrossentropy(),
            loss=self.ctc_loss,
            #metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
            metrics=[HOMRDataset.EditDistanceMetric()]
        )
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    def ctc_loss(self, gold_labels: tf.RaggedTensor, logits: tf.RaggedTensor) -> tf.Tensor:
        assert isinstance(gold_labels, tf.RaggedTensor), "Gold labels given to CTC loss must be RaggedTensors"
        assert isinstance(logits, tf.RaggedTensor), "Logits given to CTC loss must be RaggedTensors"

        # TODO: Use tf.nn.ctc_loss to compute the CTC loss.
        # - Convert the `gold_labels` to SparseTensor and pass `None` as `label_length`.
        # - Convert `logits` to a dense Tensor and then either transpose the
        #   logits to `[max_audio_length, batch, dim]` or set `logits_time_major=False`
        # - Use `logits.row_lengths()` method to obtain the `logit_length`
        # - Use the last class (the one with the highest index) as the `blank_index`.
        #
        # The `tf.nn.ctc_loss` returns a value for a single batch example, so average
        # them to produce a single value and return it.

        gold_sparse = gold_labels.to_sparse()
        gold_sparse = tf.cast(gold_sparse, dtype=tf.int32)
        logits_dense = logits.to_tensor()

        logit_len = logits.row_lengths()
        logit_len = tf.cast(logit_len, dtype=tf.int32)
        blank_index = -1

        losses = tf.nn.ctc_loss(gold_sparse, logits_dense, label_length=None, logit_length=logit_len,
                                logits_time_major=False, blank_index=blank_index)
        loss = tf.reduce_mean(losses, axis=0)
        return loss

    def ctc_decode(self, logits: tf.RaggedTensor) -> tf.RaggedTensor:
        assert isinstance(logits, tf.RaggedTensor), "Logits given to CTC predict must be RaggedTensors"

        # TODO: Run `tf.nn.ctc_greedy_decoder` or `tf.nn.ctc_beam_search_decoder`
        # to perform prediction.
        # - Convert the `logits` to a dense Tensor and then transpose them
        #   to shape `[max_audio_length, batch, dim]` using `tf.transpose`
        # - Use `logits.row_lengths()` method to obtain the `sequence_length`
        # - Convert the result of the decoded from a SparseTensor to a RaggedTensor
        #predictions = ...

        logits_dense = logits.to_tensor()
        logits_transposed = tf.transpose(logits_dense, perm=[1, 0, 2])

        seq_len = logits.row_lengths()
        seq_len = tf.cast(seq_len, dtype=tf.int32)
        ctc_dec = tf.nn.ctc_greedy_decoder(inputs=logits_transposed, sequence_length=seq_len)
        #print("\nctc_dec:", ctc_dec[0][0], ctc_dec[1], ctc_dec[0])

        #predictions = tf.RaggedTensor.from_tensor(tensor=ctc_dec[0], lengths=seq_len)
        predictions = tf.RaggedTensor.from_sparse(st_input=ctc_dec[0][0])

        assert isinstance(predictions, tf.RaggedTensor), "CTC predictions must be RaggedTensors"
        return predictions

    # We override the `train_step` method, because we do not want to
    # evaluate the training data for performance reasons
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {"loss": metric.result() for metric in self.metrics if metric.name == "loss"}

    # We override `predict_step` to run CTC decoding during prediction.
    def predict_step(self, data):
        data = data[0] if isinstance(data, tuple) else data
        y_pred = self(data, training=False)
        y_pred = self.ctc_decode(y_pred)
        return y_pred

    # We override `test_step` to run CTC decoding during evaluation.
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compute_loss(x, y, y_pred)
        y_pred = self.ctc_decode(y_pred)
        return self.compute_metrics(x, y, y_pred, None)


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

    # Load the data. The "image" is a grayscale image represented using
    # a single channel of `tf.uint8`s in [0-255] range.
    homr = HOMRDataset()

    # TODO: Create the model and train it
    #model = ...

    train = homr.train.map(lambda x: (x["image"], x["marks"]))
    dev = homr.dev.map(lambda x: (x["image"], x["marks"]))
    test = homr.test.map(lambda x: (x["image"]))

    # debug:
    if True:
        for example in train:
            #print(example[0], example[1])
            print(example[0].shape, example[1].shape)

    # for quicker debug:
    take = False
    if take:
        take_count = 13
        train = train.take(take_count)
        dev = dev.take(take_count)
        test = test.take(take_count)

    # batch up the datasets:
    #train = train.batch(args.batch_size)
    #dev = dev.batch(args.batch_size)
    #test = test.batch(args.batch_size)

    train = train.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
    dev = dev.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
    test = test.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))

    model = MargeSimpson(args)

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1)  # maybe delete

    logs = model.fit(
        train,
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=dev,
        callbacks=[tb_callback],  # maybe delete
    )


    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "homr_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the sequences of recognized marks.
        #predictions = ...
        predictions = model.predict(test)

        for sequence in predictions:
            print(" ".join(homr.MARKS[mark] for mark in sequence), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
