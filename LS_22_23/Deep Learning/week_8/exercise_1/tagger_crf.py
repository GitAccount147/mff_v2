#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import Dict
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
tf.get_logger().addFilter(lambda m: "Analyzer.lamba_check" not in m.getMessage())  # Avoid pesky warning

from morpho_dataset import MorphoDataset

# Pat a Mat:
# 3d76595a-e687-11e9-9ce9-00505601122b
# 594215cf-e687-11e9-9ce9-00505601122b

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--debug", default=True, action="store_true", help="If given, run functions eagerly.")  # False
parser.add_argument("--epochs", default=2, type=int, help="Number of epochs.")  # 5
parser.add_argument("--max_sentences", default=1000, type=int, help="Maximum number of sentences to load.")  # None
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn", default="LSTM", choices=["LSTM", "GRU"], help="RNN layer type.")
parser.add_argument("--rnn_dim", default=24, type=int, help="RNN layer dimension.")  # 64
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=128, type=int, help="Word embedding dimension.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        # Implement a one-layer RNN network. The input `words` is
        # a `RaggedTensor` of strings, each batch example being a list of words.
        words = tf.keras.layers.Input(shape=[None], dtype=tf.string, ragged=True)

        # TODO(tagger_we): Map strings in `words` to indices by using the `word_mapping` of `train.forms`.
        ind = train.forms.word_mapping(words)

        # TODO(tagger_we): Embed input words with dimensionality `args.we_dim`. Note that the `word_mapping`
        # provides a `vocabulary_size()` call returning the number of unique words in the mapping.
        inp_shape = train.forms.word_mapping.vocabulary_size()
        emb = tf.keras.layers.Embedding(input_dim=inp_shape, output_dim=args.we_dim)(ind)

        # TODO(tagger_we): Create the specified `args.rnn` RNN layer ("LSTM" or "GRU") with
        # dimension `args.rnn_dim`. The layer should produce an output for every
        # sequence element (so a 3D output). Then apply it in a bidirectional way on
        # the embedded words, **summing** the outputs of forward and backward RNNs.
        if args.rnn == "LSTM":
            rnn = tf.keras.layers.LSTM(units=args.rnn_dim, return_sequences=True)
        else:
            rnn = tf.keras.layers.GRU(units=args.rnn_dim, return_sequences=True)
        bid = tf.keras.layers.Bidirectional(layer=rnn, merge_mode='sum')
        hidden = bid(emb)

        # TODO: Add a final classification layer into as many classes as there are unique
        # tags in the `word_mapping` of `train.tags`. Note that **no activation** should
        # be used, the CRF operations will take care of it.
        #predictions = ...
        unique_train_tags = train.tags.word_mapping.vocabulary_size()
        predictions = tf.keras.layers.Dense(units=unique_train_tags)(hidden)

        # Check that the created predictions are a 3D tensor.
        assert predictions.shape.rank == 3

        super().__init__(inputs=words, outputs=predictions)

        # We compile the model with CRF loss and SpanLabelingF1 metric.
        self.compile(optimizer=tf.optimizers.Adam(jit_compile=False),
                     loss=self.crf_loss,
                     metrics=[self.SpanLabelingF1Metric(train.tags.word_mapping.get_vocabulary(), name="f1")])

        # TODO: Create `self._crf_weights`, a trainable zero-initialized tf.float32 matrix variable
        # of size [number of unique train tags, number of unique train tags], using `self.add_weight`.
        self._crf_weights = self.add_weight(shape=[unique_train_tags, unique_train_tags],
                                            dtype=tf.float32, initializer=tf.zeros_initializer)

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    def crf_loss(self, gold_labels: tf.RaggedTensor, logits: tf.RaggedTensor) -> tf.Tensor:
        assert isinstance(gold_labels, tf.RaggedTensor), "Gold labels given to CRF loss must be RaggedTensors"
        assert isinstance(logits, tf.RaggedTensor), "Logits given to CRF loss must be RaggedTensors"

        # TODO: Use `tfa.text.crf_log_likelihood` to compute the CRF log likelihood.
        # You will have to convert both logits and gold_labels to dense Tensors and
        # use `gold_labels.row_lengths()` as `sequence_lengths`. Use the `self._crf_weights`
        # as the transition weights.
        #
        # Finally, compute the loss using the computed log likelihoods, averaging the
        # individual batch examples.
        #raise NotImplementedError()

        logits_dense = logits.to_tensor()
        gold_dense = gold_labels.to_tensor()

        seq_len = gold_labels.row_lengths()
        trans_weights = self._crf_weights

        crf_log_like = tfa.text.crf_log_likelihood(inputs=logits_dense, tag_indices=gold_dense,
                                                   sequence_lengths=seq_len, transition_params=trans_weights)
        loss = - tf.reduce_mean(crf_log_like[0], axis=0)
        return loss

    def crf_decode(self, logits: tf.RaggedTensor) -> tf.RaggedTensor:
        assert isinstance(logits, tf.RaggedTensor), "Logits given to CRF decoding must be RaggedTensors"

        # TODO: Perform CRF decoding using `tfa.text.crf_decode`. Convert the
        # logits analogously as in `crf_loss`. Finally, convert the result
        # to a ragged tensor.
        #predictions = ...

        logits_dense = logits.to_tensor()

        trans_weights = self._crf_weights
        seq_len = logits.row_lengths()

        crf_dec = tfa.text.crf_decode(potentials=logits_dense, transition_params=trans_weights, sequence_length=seq_len)
        predictions = tf.RaggedTensor.from_tensor(tensor=crf_dec[0], lengths=seq_len)

        assert isinstance(predictions, tf.RaggedTensor)
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

    # We override `predict_step` to run CRF decoding during prediction.
    def predict_step(self, data):
        data = data[0] if isinstance(data, tuple) else data
        y_pred = self(data, training=False)
        y_pred = self.crf_decode(y_pred)
        return y_pred

    # We override `test_step` to run CRF decoding during evaluation.
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compute_loss(x, y, y_pred)
        y_pred = self.crf_decode(y_pred)
        return self.compute_metrics(x, y, y_pred, None)

    class SpanLabelingF1Metric(tf.metrics.Metric):
        """Keras-like metric evaluating span labeling F1-score of RaggedTensors."""
        def __init__(self, tags, name="span_labeling_f1", dtype=None):
            super().__init__(name, dtype)
            self._tags = tags
            self._counts = self.add_weight("counts", shape=[3], initializer=tf.initializers.Zeros(), dtype=tf.int64)

        def reset_state(self):
            self._counts.assign([0] * 3)

        def classify_spans(self, y_true, y_pred, sentence_limits):
            sentence_limits = set(sentence_limits)
            spans_true, spans_pred = set(), set()
            for spans, labels in [(spans_true, y_true), (spans_pred, y_pred)]:
                span = None
                for i, label in enumerate(self._tags[label] for label in labels):
                    if span and (label.startswith(("O", "B")) or i in sentence_limits):
                        spans.add((start, i, span))
                        span = None
                    if label.startswith("B"):
                        span, start = label[2:], i
                if span:
                    spans.add((start, len(labels), span))
            return np.array([len(spans_true & spans_pred), len(spans_pred - spans_true),
                             len(spans_true - spans_pred)], np.int64)

        def update_state(self, y, y_pred, sample_weight=None):
            assert isinstance(y, tf.RaggedTensor) and isinstance(y_pred, tf.RaggedTensor)
            assert sample_weight is None, "sample_weight currently not supported"
            counts = tf.numpy_function(self.classify_spans, (y.values, y_pred.values, y.row_limits()), tf.int64)
            self._counts.assign_add(counts)

        def result(self):
            tp, fp, fn = self._counts[0], self._counts[1], self._counts[2]
            return tf.math.divide_no_nan(tf.cast(2 * tp, tf.float32), tf.cast(2 * tp + fp + fn, tf.float32))


def main(args: argparse.Namespace) -> Dict[str, float]:
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
    morpho = MorphoDataset("czech_cnec", max_sentences=args.max_sentences)

    # Create the model and train
    model = Model(args, morpho.train)

    # TODO(tagger_we): Construct the data for the model, each consisting of the following pair:
    # - a tensor of string words (forms) as input,
    # - a tensor of integer tag ids as targets.
    # To create the tag ids, use the `word_mapping` of `morpho.train.tags`.
    def extract_tagging_data(example):
        target = morpho.train.tags.word_mapping(example["tags"])
        inp = example["forms"]
        return inp, target
        #raise NotImplementedError()

    def create_dataset(name):
        dataset = getattr(morpho, name).dataset
        dataset = dataset.map(extract_tagging_data)
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    train, dev = create_dataset("train"), create_dataset("dev")

    logs = model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[model.tb_callback])

    # Return all metrics for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items()}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
