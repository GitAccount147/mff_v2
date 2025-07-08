#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import Dict
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from morpho_dataset import MorphoDataset

# "Bez motivace nejsou koláče. Nemáme koláče."
# 3d76595a-e687-11e9-9ce9-00505601122b
# 594215cf-e687-11e9-9ce9-00505601122b

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--debug", default=True, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")  # 5
parser.add_argument("--max_sentences", default=800, type=int, help="Maximum number of sentences to load.")  # None
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--transformer_dropout", default=0.1, type=float, help="Transformer dropout.")  # 0.
parser.add_argument("--transformer_expansion", default=4, type=float, help="Transformer FFN expansion factor.")
parser.add_argument("--transformer_heads", default=4, type=int, help="Transformer heads.")  # 4
parser.add_argument("--transformer_layers", default=2, type=int, help="Transformer layers.")  # 2
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(tf.keras.Model):
    class FFN(tf.keras.layers.Layer):
        def __init__(self, dim, expansion, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dim, self.expansion = dim, expansion
            # TODO: Create the required layers -- first a ReLU-activated dense
            # layer with `dim * expansion` units, followed by a dense layer
            # with `dim` units without an activation.
            #raise NotImplementedError()
            self.dense_activated = tf.keras.layers.Dense(units=(dim * expansion), activation=tf.nn.relu)
            self.dense = tf.keras.layers.Dense(units=dim)

        def get_config(self):
            return {"dim": self.dim, "expansion": self.expansion}

        def call(self, inputs):
            # TODO: Execute the FFN Transformer layer.
            #raise NotImplementedError()
            dense_activated = self.dense_activated(inputs)
            return self.dense(dense_activated)

    class SelfAttention(tf.keras.layers.Layer):
        def __init__(self, dim, heads, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dim, self.heads = dim, heads
            # TODO: Create weight matrices W_Q, W_K, W_V, and W_O using `self.add_weight`,
            # each with shape `[dim, dim]`; keep the default for other `add_weight` arguments
            # (which means trainable float32 matrices initialized with `"glorot_uniform"`).
            #raise NotImplementedError()
            self._W_Q = self.add_weight(shape=[dim, dim])
            self._W_K = self.add_weight(shape=[dim, dim])
            self._W_V = self.add_weight(shape=[dim, dim])
            self._W_O = self.add_weight(shape=[dim, dim])

        def get_config(self):
            return {"dim": self.dim, "heads": self.heads}

        def call(self, inputs, mask):
            # TODO: Execute the self-attention layer.
            #
            # Start by computing Q, K and V. In all cases:
            # - first multiply `inputs` by the corresponding weight matrix W_Q/W_K/W_V,
            # - reshape via `tf.reshape` to `[batch_size, max_sentence_len, heads, dim // heads]`,
            # - transpose via `tf.transpose` to `[batch_size, heads, max_sentence_len, dim // heads]`.
            Q = inputs @ self._W_Q
            K = inputs @ self._W_K
            V = inputs @ self._W_V

            #batch_size = tf.shape(inputs)[0]
            #max_sentence_len = tf.shape(inputs)[1]
            batch_size = tf.shape(Q)[0]
            max_sentence_len = tf.shape(Q)[1]
            dim = self.dim
            heads = self.heads
            Q = tf.reshape(Q, shape=[batch_size, max_sentence_len, heads, dim // heads])
            K = tf.reshape(K, shape=[batch_size, max_sentence_len, heads, dim // heads])
            V = tf.reshape(V, shape=[batch_size, max_sentence_len, heads, dim // heads])

            perm = [0, 2, 1, 3]
            Q = tf.transpose(Q, perm=perm)
            K = tf.transpose(K, perm=perm)
            V = tf.transpose(V, perm=perm)

            # TODO: Continue by computing the self-attention weights as Q @ K^T,
            # normalizing by the square root of `dim // heads`.

            perm = [0, 1, 3, 2]
            #perm = [0, 1, 2, 3]
            sqrt = tf.math.sqrt(tf.cast(dim // heads, dtype=tf.float32))
            attention_weights = (Q @ tf.transpose(K, perm=perm) / sqrt)  # divide not multiply
            #attention_weights = (Q @ K * sqrt)
            #print(tf.shape(Q), tf.shape(K))
            #print(Q.shape, K.shape)

            # TODO: Apply the softmax, but including a suitable mask ignoring all padding words.
            # The original `mask` is a bool matrix of shape `[batch_size, max_sentence_len]`
            # indicating which words are valid (`True`) or padding (`False`).
            # To mask an input to softmax, replace it by -1e9 (theoretically we should use
            # minus infinity, but `tf.math.exp(-1e9)` is also zero because of limited precision).
            small = tf.ones(shape=tf.shape(attention_weights)) * (-1e9)
            #print(small.shape, mask.shape)
            #cond = tf.transpose(tf.expand_dims(tf.expand_dims(mask, axis=-1), axis=-1), perm=[0, 2, 1, 3])
            cond = tf.transpose(tf.expand_dims(tf.expand_dims(mask, axis=-1), axis=-1), perm=[0, 2, 3, 1])
            #cond = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=-1)
            masked_inputs = tf.where(condition=cond, x=attention_weights, y=small)
            softmax = tf.nn.softmax(masked_inputs)

            # TODO: Finally,
            # - take a weighted combination of values V according to the computed attention
            #   (using a suitable matrix multiplication),
            # - transpose the result to `[batch_size, max_sentence_len, heads, dim // heads]`,
            # - reshape to `[batch_size, max_sentence_len, dim]`,
            # - multiply the result by the W_O matrix.
            #raise NotImplementedError()

            #print(softmax.shape, V.shape)
            attention = softmax @ V
            #attention = V @ softmax
            #attention = tf.transpose(softmax, perm=[0, 1, 3, 2]) @ V
            #attention = softmax * V


            perm = [0, 2, 1, 3]
            attention = tf.transpose(attention, perm=perm)
            attention = tf.reshape(attention, shape=[batch_size, max_sentence_len, dim])
            result = attention @ self._W_O
            return result

    class PositionalEmbedding(tf.keras.layers.Layer):
        def __init__(self, dim, *args, **kwargs):
            assert dim % 2 == 0  # The `dim` needs to be even to have the same number of sin&cos.
            super().__init__(*args, **kwargs)
            self.dim = dim

        def get_config(self):
            return {"dim": self.dim}

        def call_old(self, inputs):
            # TODO: Compute the sinusoidal positional embeddings.
            # They have a shape `[max_sentence_len, self.dim]`, where `self.dim` is even and
            # - for `0 <= i < dim / 2`, the value on index `[pos, i]` should be
            #     `sin(pos / 10_000 ** (2 * i / dim))`
            # - the value on index `[pos, i]` for `i >= dim / 2` should be
            #     `cos(pos / 10_000 ** (2 * (i - dim/2) / dim))`
            # - the `0 <= pos < max_sentence_len` is the sentence index.
            # This order is the same as in the visualization on the slides, but
            # different from the original paper where `sin` and `cos` interleave.
            #raise NotImplementedError()

            #_, max_sentence_len, dim = tf.shape(inputs)
            max_sentence_len = tf.shape(inputs)[1]
            dim = tf.shape(inputs)[2]

            #dim_half = tf.cast(dim / 2, dtype=tf.int32)
            pos_matrix_sin = tf.tile(tf.range(0, max_sentence_len), multiples=[dim / 2])
            pos_matrix_cos = tf.tile(tf.range(0, max_sentence_len), multiples=[dim / 2])
            pos_matrix_sin = tf.reshape(pos_matrix_sin, shape=[max_sentence_len, dim // 2])
            pos_matrix_cos = tf.reshape(pos_matrix_cos, shape=[max_sentence_len, dim // 2])  # kinda useless to make twice but w/e
            pos_matrix_sin = tf.cast(pos_matrix_sin, dtype=tf.float32)
            pos_matrix_cos = tf.cast(pos_matrix_cos, dtype=tf.float32)

            i_matrix_sin = tf.tile(tf.range(0, dim / 2), multiples=[max_sentence_len])
            i_matrix_cos = tf.tile(tf.range(dim / 2, dim), multiples=[max_sentence_len])
            i_matrix_sin = tf.reshape(i_matrix_sin, shape=[max_sentence_len, dim // 2])
            i_matrix_cos = tf.reshape(i_matrix_cos, shape=[max_sentence_len, dim // 2])
            i_matrix_sin = tf.cast(i_matrix_sin, dtype=tf.float32)
            i_matrix_cos = tf.cast(i_matrix_cos, dtype=tf.float32)
            #print(i_matrix_cos, pos_matrix_cos)

            #sines = tf.sin(pos_matrix[:, :(dim / 2)] / 10000 ** (2 * i_matrix[:, :(dim / 2)] / dim))
            #cosines = tf.cos(pos_matrix[:, (dim / 2):] / 10000 ** (2 * (i_matrix[:, (dim / 2):] - dim / 2) / dim))
            dim_float = tf.cast(dim, dtype=tf.float32)
            sines = tf.sin(pos_matrix_sin / (10000 ** (2 * i_matrix_sin / dim_float)))
            cosines = tf.cos(pos_matrix_cos / (10000 ** (2 * (i_matrix_cos - dim_float / 2) / dim_float)))

            concat = tf.concat([sines, cosines], axis=-1)
            return concat

        def call(self, inputs):
            # TODO: Compute the sinusoidal positional embeddings.
            # They have a shape `[max_sentence_len, self.dim]`, where `self.dim` is even and
            # - for `0 <= i < dim / 2`, the value on index `[pos, i]` should be
            #     `sin(pos / 10_000 ** (2 * i / dim))`
            # - the value on index `[pos, i]` for `i >= dim / 2` should be
            #     `cos(pos / 10_000 ** (2 * (i - dim/2) / dim))`
            # - the `0 <= pos < max_sentence_len` is the sentence index.
            # This order is the same as in the visualization on the slides, but
            # different from the original paper where `sin` and `cos` interleave.
            #raise NotImplementedError()

            #_, max_sentence_len, dim = tf.shape(inputs)
            max_sentence_len = tf.shape(inputs)[1]
            dim = tf.shape(inputs)[2]

            #dim_half = tf.cast(dim / 2, dtype=tf.int32)
            #print("max_sen_len:", max_sentence_len, "dim:", dim)
            pos_matrix_sin = tf.tile(tf.range(0, max_sentence_len), multiples=[dim / 2])
            pos_matrix_cos = tf.tile(tf.range(0, max_sentence_len), multiples=[dim / 2])
            #print(pos_matrix_sin)
            #pos_matrix_sin = tf.reshape(pos_matrix_sin, shape=[max_sentence_len, dim // 2])
            #pos_matrix_cos = tf.reshape(pos_matrix_cos, shape=[max_sentence_len, dim // 2])  # kinda useless to make twice but w/e
            pos_matrix_sin = tf.reshape(pos_matrix_sin, shape=[dim // 2, max_sentence_len])
            pos_matrix_cos = tf.reshape(pos_matrix_cos, shape=[dim // 2, max_sentence_len])  # kinda useless to make twice but w/e
            pos_matrix_sin = tf.transpose(pos_matrix_sin)
            pos_matrix_cos = tf.transpose(pos_matrix_cos)
            pos_matrix_sin = tf.cast(pos_matrix_sin, dtype=tf.float32)
            pos_matrix_cos = tf.cast(pos_matrix_cos, dtype=tf.float32)
            #print(pos_matrix_sin)

            i_matrix_sin = tf.tile(tf.range(0, dim / 2), multiples=[max_sentence_len])
            i_matrix_cos = tf.tile(tf.range(dim / 2, dim), multiples=[max_sentence_len])
            #print(i_matrix_sin)
            i_matrix_sin = tf.reshape(i_matrix_sin, shape=[max_sentence_len, dim // 2])
            i_matrix_cos = tf.reshape(i_matrix_cos, shape=[max_sentence_len, dim // 2])
            i_matrix_sin = tf.cast(i_matrix_sin, dtype=tf.float32)
            i_matrix_cos = tf.cast(i_matrix_cos, dtype=tf.float32)
            #print(i_matrix_sin)

            #sines = tf.sin(pos_matrix[:, :(dim / 2)] / 10000 ** (2 * i_matrix[:, :(dim / 2)] / dim))
            #cosines = tf.cos(pos_matrix[:, (dim / 2):] / 10000 ** (2 * (i_matrix[:, (dim / 2):] - dim / 2) / dim))
            dim_float = tf.cast(dim, dtype=tf.float32)
            sines = tf.sin(pos_matrix_sin / (10000 ** (2 * i_matrix_sin / dim_float)))
            cosines = tf.cos(pos_matrix_cos / (10000 ** (2 * (i_matrix_cos - dim_float / 2) / dim_float)))

            concat = tf.concat([sines, cosines], axis=-1)
            return concat

    class Transformer(tf.keras.layers.Layer):
        def __init__(self, layers, dim, expansion, heads, dropout, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.layers, self.dim, self.expansion, self.heads, self.dropout = layers, dim, expansion, heads, dropout
            # TODO: Create:
            # - the positional embedding layer;
            # - the required number of transformer layers, each consisting of
            #   - a layer normalization and a self-attention layer followed by a dropout layer,
            #   - a layer normalization and a FFN layer followed by a dropout layer.
            self.embedding_layer = Model.PositionalEmbedding(dim)
            self.layers_list = []
            for i in range(layers):
                layer_norm_1 = tf.keras.layers.LayerNormalization()
                self_attention = Model.SelfAttention(dim=dim, heads=heads)
                drop_1 = tf.keras.layers.Dropout(rate=dropout)
                layer_norm_2 = tf.keras.layers.LayerNormalization()
                ffn_layer = Model.FFN(dim=dim, expansion=expansion)
                drop_2 = tf.keras.layers.Dropout(rate=dropout)
                self.layers_list.append([layer_norm_1, self_attention, drop_1, layer_norm_2, ffn_layer, drop_2])
                # we probably dont need the duplicates

        def get_config(self):
            return {name: getattr(self, name) for name in ["layers", "dim", "expansion", "heads", "dropout"]}

        def call(self, inputs, mask):
            # TODO: First compute the positional embeddings.
            embeddings = self.embedding_layer(inputs)

            # TODO: Add the positional embeddings to the `inputs` and then
            # perform the given number of transformer layers, composed of
            # - a self-attention sub-layer, followed by
            # - a FFN sub-layer.
            # In each sub-layer, pass the input through LayerNorm, then compute
            # the corresponding operation, apply dropout, and finally add this result
            # to the original sub-layer input. Note that the given `mask` should be
            # passed to the self-attention operation to ignore the padding words.
            #raise NotImplementedError()

            result = inputs + embeddings
            for i in range(len(self.layers_list)):
                layer_norm_1, self_attention, drop_1, layer_norm_2, ffn_layer, drop_2 = self.layers_list[i]

                # self-attention sub-layer:
                sub_layer_1 = layer_norm_1(result)
                sub_layer_1 = self_attention(inputs=sub_layer_1, mask=mask)
                sub_layer_1 = drop_1(sub_layer_1)
                result += sub_layer_1

                # FFN sub-layer:
                sub_layer_2 = layer_norm_2(result)
                sub_layer_2 = ffn_layer(sub_layer_2)
                sub_layer_2 = drop_2(sub_layer_2)
                result += sub_layer_2

            return result

    def __init__(self, args, train):
        # Implement a transformer encoder network. The input `words` is
        # a RaggedTensor of strings, each batch example being a list of words.
        words = tf.keras.layers.Input(shape=[None], dtype=tf.string, ragged=True)

        # TODO(tagger_we): Map strings in `words` to indices by using the `word_mapping` of `train.forms`.
        ind = train.forms.word_mapping(words)

        # TODO(tagger_we): Embed input words with dimensionality `args.we_dim`. Note that the `word_mapping`
        # provides a `vocabulary_size()` call returning the number of unique words in the mapping.
        inp_shape = train.forms.word_mapping.vocabulary_size()
        emb = tf.keras.layers.Embedding(input_dim=inp_shape, output_dim=args.we_dim)(ind)

        # TODO: Call the Transformer layer:
        # - create a `Model.Transformer` layer, using suitable options from `args`
        #   (using `args.we_dim` for the `dim` argument),
        # - when calling the layer, convert the ragged tensor with the input words embedding
        #   to a dense one, and also pass the following argument as a mask:
        #     `mask=tf.sequence_mask(ragged_tensor_with_input_words_embeddings.row_lengths())`
        # - finally, convert the result back to a ragged tensor.
        transformer = Model.Transformer(layers=args.transformer_layers, dim=args.we_dim,
                                        expansion=args.transformer_expansion, heads=args.transformer_heads,
                                        dropout=args.transformer_dropout)
        row_lengths = emb.row_lengths()
        mask = tf.sequence_mask(lengths=row_lengths)
        dense = emb.to_tensor()
        hidden = transformer(inputs=dense, mask=mask)
        ragged = tf.RaggedTensor.from_tensor(tensor=hidden, lengths=row_lengths)

        # TODO(tagger_we): Add a softmax classification layer into as many classes as there are unique
        # tags in the `word_mapping` of `train.tags`. Note that the Dense layer can process
        # a `RaggedTensor` without any problem.
        #predictions = ...
        units = train.tags.word_mapping.vocabulary_size()
        predictions = tf.keras.layers.Dense(units=units, activation=tf.nn.softmax)(ragged)

        # Check that the created predictions are a 3D tensor.
        assert predictions.shape.rank == 3

        super().__init__(inputs=words, outputs=predictions)

        def ragged_sparse_categorical_crossentropy(y_true, y_pred):
            return tf.losses.SparseCategoricalCrossentropy()(y_true.values, y_pred.values)

        self.compile(optimizer=tf.optimizers.Adam(jit_compile=False),
                     loss=ragged_sparse_categorical_crossentropy,
                     metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")])

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)


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
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    # Create the model and train
    model = Model(args, morpho.train)

    # TODO(tagger_we): Construct the data for the model, each consisting of the following pair:
    # - a tensor of string words (forms) as input,
    # - a tensor of integer tag ids as targets.
    # To create the tag ids, use the `word_mapping` of `morpho.train.tags`.
    def extract_tagging_data(example):
        #raise NotImplementedError()
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

    # Return development and training losses for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items() if "loss" in metric}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
