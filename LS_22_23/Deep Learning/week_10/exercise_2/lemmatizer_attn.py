#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import Dict
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

# "Bez motivace nejsou koláče. Nemáme koláče."
# 3d76595a-e687-11e9-9ce9-00505601122b
# 594215cf-e687-11e9-9ce9-00505601122b

import numpy as np
import tensorflow as tf
tf.get_logger().addFilter(lambda m: "Analyzer.lamba_check" not in m.getMessage())  # Avoid pesky warning

from morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=2, type=int, help="Batch size.")  # 10
parser.add_argument("--cle_dim", default=32, type=int, help="CLE embedding dimension.")  # 64
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=500, type=int, help="Maximum number of sentences to load.")  # None
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn_dim", default=32, type=int, help="RNN layer dimension.")  # 64
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--tie_embeddings", default=False, action="store_true", help="Tie target embeddings.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


class WithAttention(tf.keras.layers.AbstractRNNCell):
    """A class adding Bahdanau attention to the given RNN cell."""
    def __init__(self, cell, attention_dim):
        super().__init__()
        self._cell = cell

        # TODO: Define
        # - `self._project_encoder_layer` as a Dense layer with `attention_dim` outputs
        # - `self._project_decoder_layer` as a Dense layer with `attention_dim` outputs
        # - `self._output_layer` as a Dense layer with 1 output

        self._project_encoder_layer = tf.keras.layers.Dense(units=attention_dim)
        self._project_decoder_layer = tf.keras.layers.Dense(units=attention_dim)
        self._output_layer = tf.keras.layers.Dense(units=1)

    @property
    def state_size(self):
        return self._cell.state_size

    def setup_memory(self, encoded):
        self._encoded = encoded
        # TODO: Pass the `encoded` through the `self._project_encoder_layer` and store
        # the result as `self._encoded_projected`.
        #self._encoded_projected = ...
        self._encoded_projected = self._project_encoder_layer(encoded)

    def call(self, inputs, states):
        # TODO: Compute the attention.
        # - According to the definition, we need to project the encoder states, but we have
        #   already done that in `setup_memory`, so we just take `self._encoded_projected`.
        # - Compute projected decoder state by passing the given state through the `self._project_decoder_layer`.
        # - Sum the two projections. However, the first has shape `[batch_size, input_sequence_len, attention_dim]`
        #   and the second just `[batch_size, attention_dim]`. Therefore, expand the second projection
        #   to `[batch_size, 1, attention_dim]`, and then broadcasting will allow to sum the two projections.
        # - Pass the sum through the `tf.nn.tanh` and then through the `self._output_layer`.
        # - Then, run softmax on a suitable axis, generating `weights`.
        # - Multiply the original (non-projected) encoder states `self._encoded` with `weights` and sum
        #   the result in the axis corresponding to characters, generating `attention`. Therefore,
        #   `attention` is a fixed-size representation for every batch element, independently on
        #   how many characters the corresponding input form had.
        # - Finally, concatenate `inputs` and `attention` (in this order), and call the `self._cell`
        #   on this concatenated input and the `states`, returning the result.
        proj_dec_state = self._project_decoder_layer(states[0])
        shapes = tf.shape(proj_dec_state)
        proj_dec_state = tf.reshape(proj_dec_state, shape=[shapes[0], 1, shapes[1]])
        our_sum = self._encoded_projected + proj_dec_state
        our_sum = tf.nn.tanh(our_sum)
        our_sum = self._output_layer(our_sum)
        weights = tf.nn.softmax(our_sum, axis=1)
        attention = tf.reduce_sum(self._encoded * weights, axis=1)
        conc = tf.concat([inputs, attention], axis=1)

        return self._cell(conc, states)


class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()

        self._source_mapping = train.forms.char_mapping
        self._target_mapping = train.lemmas.char_mapping
        self._target_mapping_inverse = type(self._target_mapping)(
            vocabulary=self._target_mapping.get_vocabulary(), invert=True)

        # TODO(lemmatizer_noattn): Define
        # - `self._source_embedding` as an embedding layer of source ids into `args.cle_dim` dimensions
        #self._source_embedding = ...
        units = self._source_mapping.vocabulary_size()
        self._source_embedding = tf.keras.layers.Embedding(input_dim=units, output_dim=args.cle_dim)

        # TODO: Define
        # - `self._source_rnn` as a bidirectional GRU with `args.rnn_dim` units, returning **whole sequences**,
        #   summing opposite directions
        #self._source_rnn = ...
        gru = tf.keras.layers.GRU(units=args.rnn_dim, return_sequences=True)
        self._source_rnn = tf.keras.layers.Bidirectional(gru, merge_mode="sum")

        # TODO: Then define
        # - `self._target_rnn` as a `tf.keras.layers.RNN` returning whole sequences, utilizing the
        #   attention-enhanced cell using `WithAttention` with `attention_dim` of `args.rnn_dim`,
        #   employing the `tf.keras.layers.GRUCell` with `args.rnn_dim` units as the underlying cell.
        #self._target_rnn = ...
        gru2 = tf.keras.layers.GRUCell(units=args.rnn_dim)
        self._target_rnn = tf.keras.layers.RNN(cell=WithAttention(cell=gru2, attention_dim=args.rnn_dim), return_sequences=True)

        # TODO(lemmatizer_noattn): Then define
        # - `self._target_output_layer` as a Dense layer into as many outputs as there are unique target chars
        #self._target_output_layer = ...

        units = self._target_mapping.vocabulary_size()
        self._target_output_layer = tf.keras.layers.Dense(units=units)

        if not args.tie_embeddings:
            # TODO(lemmatizer_noattn): Define the `self._target_embedding` as an embedding layer of the target
            # ids into `args.cle_dim` dimensions.
            #self._target_embedding = ...

            self._target_embedding = tf.keras.layers.Embedding(input_dim=units, output_dim=args.cle_dim)
        else:
            self._target_output_layer.build(args.rnn_dim)
            # TODO(lemmatizer_noattn): Create a function `self._target_embedding` which computes the embedding of given
            # target ids. When called, use `tf.gather` to index the transposition of the shared embedding
            # matrix `self._target_output_layer.kernel` multiplied by the square root of `args.rnn_dim`.
            #self._target_embedding = ...

            def manual_target_embedding(id):
                scalar = tf.math.sqrt(tf.cast(args.rnn_dim, dtype=tf.float32))
                matrix = tf.transpose(self._target_output_layer.kernel) * scalar
                # ?!?!?!??!?!?!?!?!?!?!??!?!?!??!?!?!??!????!?! divide not multiply?
                result = tf.gather(matrix, id, axis=0)
                return result

            self._target_embedding = manual_target_embedding

        # Compile the model
        self.compile(
            optimizer=tf.optimizers.Adam(jit_compile=False),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.Accuracy(name="accuracy")],
        )

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    def encoder(self, inputs: tf.Tensor) -> tf.Tensor:
        # TODO(lemmatizer_noattn): Embed the inputs using `self._source_embedding`.

        emb = self._source_embedding(inputs)

        # TODO: Run the `self._source_rnn` on the embedded sequences, then convert its result
        # to a dense tensor using the `.to_tensor()` call, and return it.
        src = self._source_rnn(emb)
        return src.to_tensor()

    def decoder_training(self, encoded: tf.Tensor, targets: tf.Tensor) -> tf.Tensor:
        # TODO(lemmatizer_noattn): Generate inputs for the decoder, which is obtained from `targets` by
        # - prepending `MorphoDataset.BOW` as the first element of every batch example,
        # - dropping the last element of `targets` (which is `MorphoDataset.EOW`)

        shape = tf.shape(targets)[0]
        bow_tensor = tf.ones(shape=(shape, 1), dtype=tf.int64) * MorphoDataset.BOW
        prep = tf.concat([bow_tensor, targets], axis=-1)
        drop = prep[:, :-1]

        # TODO: Pre-compute the projected encoder states in the attention by calling
        # the `setup_memory` of the `self._target_rnn.cell` on the `encoded` input.
        #print(self._target_rnn.cell)
        self._target_rnn.cell.setup_memory(encoded)
        #encoded = self._target_rnn.cell._encoded_projected

        # TODO: Process the generated inputs by
        # - the `self._target_embedding` layer to obtain embeddings,
        # - the `self._target_rnn` layer, passing an additional parameter `initial_state=[encoded[:, 0]]`,
        # - the `self._target_output_layer` to obtain logits,
        # and return the result.

        emb = self._target_embedding(drop)
        emb = self._target_rnn(emb, initial_state=[encoded[:, 0]])
        return self._target_output_layer(emb)

    @tf.function
    def decoder_inference(self, encoded: tf.Tensor, max_length: tf.Tensor) -> tf.Tensor:
        """The decoder_inference runs a while-cycle inside a computation graph.

        To that end, it needs to be explicitly marked as @tf.function, so that the
        below `while` cycle is "embedded" in the computation graph. Alternatively,
        we might explicitly use the `tf.while_loop` operation, but a native while
        cycle is more readable.
        """
        batch_size = tf.shape(encoded)[0]
        max_length = tf.cast(max_length, tf.int32)

        # TODO(decoder_training): Pre-compute the projected encoder states in the attention by calling
        # the `setup_memory` of the `self._target_rnn.cell` on the `encoded` input.
        self._target_rnn.cell.setup_memory(encoded)
        #encoded = self._target_rnn.cell._encoded_projected

        # TODO: Define the following variables, that we will use in the cycle:
        # - `index`: a scalar tensor with dtype `tf.int32` initialized to 0,
        # - `inputs`: a batch of `MorphoDataset.BOW` symbols of type `tf.int64`,
        # - `states`: initial RNN state from the encoder, i.e., `[encoded[:, 0]]`,
        #index = ...
        #inputs = ...
        #states = ...
        index = tf.constant(0, dtype=tf.int32)
        inputs = tf.ones(shape=batch_size, dtype=tf.int64) * MorphoDataset.BOW
        states = [encoded[:, 0]]

        # We collect the results from the while-cycle into the following `tf.TensorArray`,
        # which is a dynamic collection of tensors that can be written to. We also
        # create `result_lengths` containing lengths of completely generated sequences,
        # starting with `max_length` and optionally decreasing when an EOW is generated.
        result = tf.TensorArray(tf.int64, size=max_length)
        result_lengths = tf.fill([batch_size], max_length)

        while tf.math.logical_and(index < max_length, tf.math.reduce_any(result_lengths == max_length)):
            # TODO(lemmatizer_noattn):
            # - First embed the `inputs` using the `self._target_embedding` layer.
            # - Then call `self._target_rnn.cell` using two arguments, the embedded `inputs`
            #   and the current `states`. The call returns a pair of (outputs, new states),
            #   where the new states should replace the current `states`.
            # - Pass the outputs through the `self._target_output_layer`.
            # - Finally generate the most probable prediction for every batch example.
            #predictions = ...

            inputs = self._target_embedding(inputs)
            outputs, new_states = self._target_rnn.cell(inputs, states)
            states = new_states
            out = self._target_output_layer(outputs)

            predictions = tf.argmax(out, axis=-1)

            # Store the predictions in the `result` on the current `index`. Then update
            # the `result_lengths` by setting it to current `index` if an EOW was generated
            # for the first time.
            result = result.write(index, predictions)
            result_lengths = tf.where(
                tf.math.logical_and(predictions == MorphoDataset.EOW, result_lengths > index), index, result_lengths)

            # TODO(lemmatizer_noattn): Finally,
            # - set `inputs` to the `predictions`,
            # - increment the `index` by one.
            #inputs = ...
            #index = ...

            inputs = predictions
            index += 1

        # Stack the `result` into a dense rectangular tensor, and create a ragged tensor
        # from it using the `result_lengths`.
        result = tf.RaggedTensor.from_tensor(tf.transpose(result.stack()), lengths=result_lengths)
        return result

    def train_step(self, data):
        x, y = data

        # Forget about sentence boundaries and instead consider
        # all valid form-lemma pairs as independent batch examples.
        x_flat, y_flat = x.values, y.values

        # TODO(lemmatizer_noattn): Process `x_flat` by
        # - `tf.strings.unicode_split` with encoding "UTF-8" to generate a ragged
        #   tensor with individual characters as strings,
        # - `self._source_mapping` to remap the character strings to ids.
        #x_flat = ...

        x_flat = tf.strings.unicode_split(input=x_flat, input_encoding="UTF-8")
        x_flat = self._source_mapping(x_flat)

        # TODO(lemmatizer_noattn): Process `y_flat` by
        # - `tf.strings.unicode_split` with encoding "UTF-8" to generate a ragged
        #   tensor with individual characters as strings,
        # - `self._target_mapping` to remap the character strings to ids,
        # - finally, append a `MorphoDataset.EOW` to the end of every batch example.
        #y_flat = ...

        y_flat = tf.strings.unicode_split(input=y_flat, input_encoding="UTF-8")
        y_flat = self._target_mapping(y_flat)
        shape = tf.shape(x_flat)[0]
        eow_tensor = tf.ones(shape=(shape, 1), dtype=tf.int64) * MorphoDataset.EOW

        y_flat = tf.concat([y_flat, eow_tensor], axis=-1)

        with tf.GradientTape() as tape:
            encoded = self.encoder(x_flat)
            y_pred = self.decoder_training(encoded, y_flat)
            loss = self.compute_loss(x, y_flat.values, y_pred.values)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        return {"loss": metric.result() for metric in self.metrics if metric.name == "loss"}

    def predict_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        # As in `train_step`, forget about sentence boundaries.
        data_flat = data.values

        # TODO(lemmatizer_noattn): As in `train_step`, pass `data_flat` through
        # - `tf.strings.unicode_split` with encoding "UTF-8" to generate a ragged
        #   tensor with individual characters as strings,
        # - `self._source_mapping` to remap the character strings to ids.
        #data_flat = ...

        data_flat = tf.strings.unicode_split(input=data_flat, input_encoding="UTF-8")
        data_flat = self._source_mapping(data_flat)

        # bzz straka:
        encoded = self.encoder(data_flat)
        y_pred = self.decoder_inference(encoded, data_flat.bounding_shape(axis=1) + 10)
        y_pred = self._target_mapping_inverse(y_pred)
        y_pred = tf.strings.reduce_join(y_pred, axis=-1)

        # Finally, convert the individual lemmas back to sentences of lemmas using
        # the original sentence boundaries.
        y_pred = data.with_values(y_pred)
        return y_pred

    def test_step(self, data):
        x, y = data
        y_pred = self.predict_step(x)
        self.compiled_metrics.update_state(tf.ones_like(y, dtype=tf.int32), tf.cast(y_pred == y, tf.int32))
        return {m.name: m.result() for m in self.metrics if m.name != "loss"}


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
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences, add_bow_eow=True)

    # Create the model and train
    model = Model(args, morpho.train)

    # Construct dataset for lemmatizer training
    def create_dataset(name):
        dataset = getattr(morpho, name).dataset
        dataset = dataset.map(lambda example: (example["forms"], example["lemmas"]))
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        return dataset
    train, dev = create_dataset("train"), create_dataset("dev")

    # Callback showing intermediate results during training
    class ShowIntermediateResults(tf.keras.callbacks.Callback):
        def __init__(self, data):
            self._iterator = iter(data.repeat())
        def on_train_batch_end(self, batch, logs=None):
            if int(model.optimizer.iterations) % 10 == 0:
                forms, lemmas = next(self._iterator)
                print(model.optimizer.iterations.numpy(),
                      *[repr(strings[0, 0].numpy().decode("utf-8"))
                        for strings in [forms, lemmas, model.predict_on_batch(forms[:1, :1])]])

    logs = model.fit(train, epochs=args.epochs, validation_data=dev, verbose=2,
                     callbacks=[ShowIntermediateResults(dev), model.tb_callback])

    # Return all metrics for ReCodEx to validate
    return {metric: values[-1] for metric, values in logs.history.items()}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
