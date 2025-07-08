#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from morpho_analyzer import MorphoAnalyzer
from morpho_dataset import MorphoDataset

# our:
from typing import Any, Dict

# Pat a Mat:
# 3d76595a-e687-11e9-9ce9-00505601122b
# 594215cf-e687-11e9-9ce9-00505601122b

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use")

# our:
parser.add_argument("--cle_dim", default=32, type=int, help="CLE embedding dimension.")  # 32
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")  # None
parser.add_argument("--rnn", default="GRU", choices=["LSTM", "GRU"], help="RNN layer type.")
parser.add_argument("--rnn_dim", default=256, type=int, help="RNN layer dimension.")  # 64
parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")  # 64
parser.add_argument("--word_masking", default=0.1, type=float, help="Mask words with the given probability.")  # 0.0
# merge_mode?


class Model(tf.keras.Model):
    # A layer setting given rate of elements to zero.
    class MaskElements(tf.keras.layers.Layer):
        def __init__(self, rate: float) -> None:
            super().__init__()
            self._rate = rate

        def get_config(self) -> Dict[str, Any]:
            return {"rate": self._rate}

        def call(self, inputs: tf.RaggedTensor, training: bool) -> tf.RaggedTensor:
            if training:
                flat = inputs.values
                input_shape = tf.shape(flat)
                input_shape = tf.cast(input_shape, dtype=tf.int64)

                rand = tf.random.uniform(shape=input_shape)

                indices = tf.cast(tf.where(rand >= self._rate), dtype=tf.int64)
                indices_shape = tf.cast(tf.shape(indices)[0], dtype=tf.int64)

                data = tf.ones(shape=indices_shape, dtype=tf.int64)
                mask = tf.scatter_nd(indices=indices, updates=data, shape=input_shape)

                masked_input = tf.multiply(inputs.values, mask)

                ragged_result = inputs.with_values(new_values=masked_input)
                return ragged_result
            else:
                return inputs

    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        words = tf.keras.layers.Input(shape=[None], dtype=tf.string, ragged=True)

        ind = train.forms.word_mapping(words)

        hidden = self.MaskElements(rate=args.word_masking)(ind)

        inp_shape_words = train.forms.word_mapping.vocabulary_size()
        emb_words = tf.keras.layers.Embedding(input_dim=inp_shape_words, output_dim=args.we_dim)(hidden)

        unique_list, unique_indices = tf.unique(words.values)

        letters_sequences = tf.strings.unicode_split(input=unique_list, input_encoding='UTF-8')

        letters_ids = train.forms.char_mapping(letters_sequences)

        inp_shape_chars = train.forms.char_mapping.vocabulary_size()
        emb_char = tf.keras.layers.Embedding(input_dim=inp_shape_chars, output_dim=args.cle_dim)(letters_ids)

        rnn = tf.keras.layers.GRU(units=args.cle_dim)
        bid = tf.keras.layers.Bidirectional(layer=rnn, merge_mode='concat')(emb_char.to_tensor())  # emb_char

        flat_repre = tf.gather(params=bid, indices=unique_indices)

        ragged = words.with_values(new_values=flat_repre)

        hidden = tf.keras.layers.Concatenate(axis=-1)([emb_words, ragged])

        if args.rnn == "LSTM":
            rnn = tf.keras.layers.LSTM(units=args.rnn_dim, return_sequences=True)
        else:
            rnn = tf.keras.layers.GRU(units=args.rnn_dim, return_sequences=True)
        bid = tf.keras.layers.Bidirectional(layer=rnn, merge_mode='sum')

        result = bid(hidden.to_tensor())  # hidden
        result = tf.RaggedTensor.from_tensor(result, hidden.row_lengths())  # nothing
        #result = tf.keras.layers.LSTM(..., return_sequences=True)(ragged_tensor.to_tensor())
        #result = tf.RaggedTensor.from_tensor(result, ragged_tensor.row_lengths())

        units = train.tags.word_mapping.vocabulary_size()
        predictions = tf.keras.layers.Dense(units=units, activation=tf.nn.softmax)(result)  # hidden

        # Check that the created predictions are a 3D tensor.
        assert predictions.shape.rank == 3

        super().__init__(inputs=words, outputs=predictions)

        def ragged_sparse_categorical_crossentropy(y_true, y_pred):
            return tf.losses.SparseCategoricalCrossentropy()(y_true.values, y_pred.values)

        self.compile(optimizer=tf.optimizers.Adam(jit_compile=False),
                     loss=ragged_sparse_categorical_crossentropy,
                     metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")])

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)


def main(args: argparse.Namespace) -> None:
    print(args)
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

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt", max_sentences=args.max_sentences)
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    # TODO: Create the model and train it
    #model = ...
    model = Model(args, morpho.train)

    def extract_tagging_data(example):
        target = morpho.train.tags.word_mapping(example["tags"])
        inp = example["forms"]
        return inp, target

    def create_dataset(name):
        dataset = getattr(morpho, name).dataset
        dataset = dataset.map(extract_tagging_data)
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    train, dev = create_dataset("train"), create_dataset("dev")

    logs = model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[model.tb_callback])

    def create_dataset_test(name):
        dataset = getattr(morpho, name).dataset
        dataset = dataset.map(extract_tagging_data)
        #dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    test = create_dataset_test("test")

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "tagger_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the tags on the test set; update the following code
        # if you use other output structure than in tagger_we.
        #predictions = model.predict(...)
        predictions = model.predict(test)

        tag_strings = morpho.train.tags.word_mapping.get_vocabulary()
        for sentence in predictions:
            for word in np.asarray(sentence):
                print(tag_strings[np.argmax(word)], file=predictions_file)
            print(file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
