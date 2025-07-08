#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
import transformers

from text_classification_dataset import TextClassificationDataset

# "Bez motivace nejsou koláče. Nemáme koláče."
# 3d76595a-e687-11e9-9ce9-00505601122b
# 594215cf-e687-11e9-9ce9-00505601122b

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=7, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

# our:
parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")

class Model_2(tf.keras.Model):
    def __init__(self, args, train, elec, fb):
        # Implement a transformer encoder network. The input `words` is
        # a RaggedTensor of strings, each batch example being a list of words.
        #words = tf.keras.layers.Input(shape=[None], dtype=tf.string, ragged=True)
        word_ids = tf.keras.layers.Input(shape=[None], dtype=tf.int32, ragged=True)

        # TODO(tagger_we): Map strings in `words` to indices by using the `word_mapping` of `train.forms`.
        #ind = train.forms.word_mapping(words)

        # TODO(tagger_we): Embed input words with dimensionality `args.we_dim`. Note that the `word_mapping`
        # provides a `vocabulary_size()` call returning the number of unique words in the mapping.
        #inp_shape = train.forms.word_mapping.vocabulary_size()
        #emb = tf.keras.layers.Embedding(input_dim=inp_shape, output_dim=args.we_dim)(word_ids)
        emb = word_ids

        # TODO: Call the Transformer layer:
        # - create a `Model.Transformer` layer, using suitable options from `args`
        #   (using `args.we_dim` for the `dim` argument),
        # - when calling the layer, convert the ragged tensor with the input words embedding
        #   to a dense one, and also pass the following argument as a mask:
        #     `mask=tf.sequence_mask(ragged_tensor_with_input_words_embeddings.row_lengths())`
        # - finally, convert the result back to a ragged tensor.
        #transformer = Model.Transformer(layers=args.transformer_layers, dim=args.we_dim,
        #                                expansion=args.transformer_expansion, heads=args.transformer_heads,
        #                                dropout=args.transformer_dropout)
        transformer = elec
        row_lengths = emb.row_lengths()
        mask = tf.sequence_mask(lengths=row_lengths)
        dense = emb.to_tensor()
        #hidden = transformer(dense, attention_mask=mask).last_hidden_state
        hidden = transformer(dense).last_hidden_state
        ragged = tf.RaggedTensor.from_tensor(tensor=hidden, lengths=row_lengths)

        # TODO(tagger_we): Add a softmax classification layer into as many classes as there are unique
        # tags in the `word_mapping` of `train.tags`. Note that the Dense layer can process
        # a `RaggedTensor` without any problem.
        # predictions = ...
        #units = train.label_mapping.vocabulary_size()
        units = fb.train.label_mapping.vocabulary_size()
        #units = 3
        predictions = tf.keras.layers.Dense(units=units, activation=tf.nn.softmax)(ragged)

        # Check that the created predictions are a 3D tensor.
        assert predictions.shape.rank == 3

        super().__init__(inputs=word_ids, outputs=predictions)

        def ragged_sparse_categorical_crossentropy(y_true, y_pred):
            return tf.losses.SparseCategoricalCrossentropy()(y_true.values, y_pred.values)

        self.compile(optimizer=tf.optimizers.Adam(jit_compile=False),
                     #loss=ragged_sparse_categorical_crossentropy,
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                     metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")])

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

class Model(tf.keras.Model):
    def __init__(self, args, eleczech_pretrained):
        # Implement a transformer encoder network. The input `words` is
        # a RaggedTensor of strings, each batch example being a list of words.
        #words = tf.keras.layers.Input(shape=[None], dtype=tf.string, ragged=True)
        batch_ids = tf.keras.layers.Input(shape=[None], dtype=tf.int32, ragged=True)
        #batch_ids = tf.keras.layers.Input(shape=[None], dtype=tf.int32)
        print("batch_ids:", batch_ids)
        emb = batch_ids

        #ind = train.forms.word_mapping(words)

        #inp_shape = train.forms.word_mapping.vocabulary_size()
        #emb = tf.keras.layers.Embedding(input_dim=inp_shape, output_dim=args.we_dim)(ind)


        #transformer = Model.Transformer(layers=args.transformer_layers, dim=args.we_dim,
        #                                expansion=args.transformer_expansion, heads=args.transformer_heads,
        #                                dropout=args.transformer_dropout)

        row_lengths = emb.row_lengths()
        #mask = tf.sequence_mask(lengths=row_lengths)

        dense = emb.to_tensor()
        #dense = emb
        mask = tf.ones(shape=tf.shape(dense))
        #dense = emb

        #hidden = transformer(inputs=dense, mask=mask)
        #hidden = eleczech_pretrained(dense, attention_mask=mask)
        hidden = eleczech_pretrained(dense)
        #hidden = eleczech_pretrained(input_ids=dense)
        hidden = hidden.last_hidden_state

        ragged = tf.RaggedTensor.from_tensor(tensor=hidden, lengths=row_lengths)
        #ragged = hidden


        #units = train.tags.word_mapping.vocabulary_size()
        units = 3
        predictions = tf.keras.layers.Dense(units=units, activation=tf.nn.softmax)(ragged)

        # Check that the created predictions are a 3D tensor.
        assert predictions.shape.rank == 3

        super().__init__(inputs=batch_ids, outputs=predictions)

        def ragged_sparse_categorical_crossentropy(y_true, y_pred):
            y_pred_val = y_pred.values
            y_true_val = y_true.values

            return tf.losses.SparseCategoricalCrossentropy()(y_true_val, y_pred_val)

        self.compile(optimizer=tf.optimizers.Adam(jit_compile=False),
                     #loss=ragged_sparse_categorical_crossentropy,
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                     metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")])

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

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

    # Load the Electra Czech small lowercased
    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/eleczech-lc-small")
    eleczech = transformers.TFAutoModel.from_pretrained("ufal/eleczech-lc-small")

    # TODO: Load the data. Consider providing a `tokenizer` to the
    # constructor of the `TextClassificationDataset`.
    facebook = TextClassificationDataset("czech_facebook", tokenizer=tokenizer.encode)

    # TODO: Create the model and train it
    #model = ...

    """
    batch = [tokenizer.encode(sentence["documents"]) for sentence in facebook.train.dataset]
    max_length = max(len(sentence) for sentence in batch)
    batch_ids = np.zeros([len(batch), max_length], dtype=np.int32)
    batch_masks = np.zeros([len(batch), max_length], dtype=np.int32)
    for i in range(len(batch)):
        batch_ids[i, :len(batch[i])] = batch[i]
        batch_masks[i, :len(batch[i])] = 1
    result = eleczech(batch_ids, attention_mask=batch_masks)
    print(result)
    """


    #train = facebook.train.dataset.map(lambda x: (x["documents"], x["labels"], tokenizer.encode(x["tokens"])))
    #train = facebook.train.dataset.map(lambda x: (x["documents"], x["labels"], x["tokens"]))
    #dev = facebook.dev.dataset.map(lambda x: (x["documents"], x["labels"], x["tokens"]))
    #test = facebook.test.dataset.map(lambda x: (x["documents"], x["tokens"]))

    def mapping(example):
        label = ...
        if example["labels"] == '0':
            label = 0
        elif example["labels"] == 'p':
            label = 1
        else:
            label = 2
        return example["tokens"], label

    #train = facebook.train.dataset.map(lambda x: (x["tokens"], x["labels"]))
    #dev = facebook.dev.dataset.map(lambda x: (x["tokens"], x["labels"]))
    #test = facebook.test.dataset.map(lambda x: (x["tokens"]))

    #train = facebook.train.dataset.map(mapping)
    #dev = facebook.dev.dataset.map(mapping)
    #test = facebook.test.dataset.map(mapping)

    def extract_tagging_data(example):
        #raise NotImplementedError()
        target = facebook.train.label_mapping(example["labels"])
        inp = example["tokens"]
        return inp, target

    def create_dataset(name):
        dataset = getattr(facebook, name).dataset
        if name != "test":
            dataset = dataset.map(extract_tagging_data)
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    train, dev = create_dataset("train"), create_dataset("dev")
    test = create_dataset("test")

    debug = True
    if debug:
        cnt = 0
        for example in train:
            print(example)
            cnt += 1
            if cnt == 2:
                break
        for example in dev:
            #print(example)
            break
        for example in test:
            #print(example)
            break

    #train = train.batch(batch_size=args.batch_size)
    #dev = dev.batch(batch_size=args.batch_size)
    #test = test.batch(batch_size=args.batch_size)

    #model = transformers.TFAutoModel.from_pretrained("ufal/eleczech-lc-small", output_hidden_states=True)
    #model = eleczech
    #model.trainable = False
    #batch = [tokenizer.encode(sentence[0]) for sentence in train]
    #print(batch)




    #model = Model(args=args, eleczech_pretrained=eleczech)
    model = Model_2(args=args, train=train, elec=eleczech, fb=facebook)
    #optimizer = tf.optimizers.Adam()
    #model.compile(optimizer=optimizer, loss=tf.losses.SparseCategoricalCrossentropy)
    model.fit(train, epochs=args.epochs, validation_data=dev)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "sentiment_analysis.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the tags on the test set.
        #predictions = ...
        predictions = model.predict(test)

        label_strings = facebook.test.label_mapping.get_vocabulary()
        for sentence in predictions:
            print(label_strings[np.argmax(sentence)], file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
