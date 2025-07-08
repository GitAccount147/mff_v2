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
parser.add_argument("--batch_size", default=200, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")


class Model(tf.keras.Model):
    def __init__(self, args, train, elec, fb):
        inputs = tf.keras.layers.Input(shape=[None], dtype=tf.int32, ragged=True)
        #inputs = tf.keras.layers.Input(shape=[None], dtype=tf.int32)
        hidden = inputs
        #print("inputs:", inputs)

        # word embedding from tagger_we/tagger_transformer:
        #inp_shape = train.forms.word_mapping.vocabulary_size()
        #emb = tf.keras.layers.Embedding(input_dim=inp_shape, output_dim=args.we_dim)(ind)

        # row lengths:
        row_lengths = hidden.row_lengths()
        mask = tf.sequence_mask(lengths=row_lengths)
        #print("row_lengths:", row_lengths)
        #print("mask:", mask)

        # dense and transformer:
        dense = hidden.to_tensor()
        #print("dense:", dense)
        elec.trainable = False
        after_eleczech = elec(dense, attention_mask=mask)  # via bert_example.py
        last_hidden = after_eleczech.last_hidden_state
        #print("after_eleczech:", after_eleczech)
        #print("last_hidden:", last_hidden)
        hidden = last_hidden

        # back to ragged:
        ragged = hidden[:, :1]
        #ragged = hidden
        #ragged = tf.RaggedTensor.from_tensor(tensor=hidden, lengths=row_lengths)
        #print("ragged:", ragged)

        # prediction:
        #units = fb.train.label_mapping.vocabulary_size()
        #units = train.label_mapping.vocabulary_size()
        units = 3
        predictions = tf.keras.layers.Dense(units=units, activation=tf.nn.softmax)(ragged)
        #print("predictions:", predictions)

        # Check that the created predictions are a 3D tensor.
        assert predictions.shape.rank == 3

        super().__init__(inputs=inputs, outputs=predictions)

        def ragged_sparse_categorical_crossentropy(y_true, y_pred):
            y_pred_val = y_pred.values
            y_true_val = y_true.values
            return tf.losses.SparseCategoricalCrossentropy()(y_true_val, y_pred_val)

        self.compile(optimizer=tf.optimizers.Adam(jit_compile=False),
                     # loss=ragged_sparse_categorical_crossentropy,
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                     metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")])


        #self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)


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
    #facebook = TextClassificationDataset("czech_facebook")

    # TODO: Create the model and train it
    #model = ...


    # GREAT THINGS WE CAME UP WITH:

    facebook = TextClassificationDataset("czech_facebook", tokenizer=tokenizer.encode)

    def extract_tagging_data(example):
        label = facebook.train.label_mapping(example["labels"])
        tokens = example["tokens"]
        return tokens, label

    def tokenize_test_dataset(example):
        tokens = example["tokens"]
        return tokens

    def create_dataset(name):
        dataset = getattr(facebook, name).dataset
        if name != "test":  # test doesnt have labels
            dataset = dataset.map(extract_tagging_data)
        else:
            dataset = dataset.map(tokenize_test_dataset)
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    train, dev = create_dataset("train"), create_dataset("dev")
    test = create_dataset("test")

    print_examples = 0
    if print_examples:
        cnt = 0
        for example in test:
            #print(example)
            #print(type(example[0]))
            #print(type(example[0]) == 'tensorflow.python.ops.ragged.ragged_tensor.RaggedTensor')
            #print(isinstance(example[0], tf.RaggedTensor))
            #print(type(example[0]) == tf.RaggedTensor)
            if not isinstance(example[0], tf.RaggedTensor):
                print("TADY TADY TADY")
            cnt += 1
            if cnt == print_examples:
                break

    model = Model(args=args, train=train, elec=eleczech, fb=facebook)



    model.fit(train, epochs=args.epochs, validation_data=dev)
    #model.fit(train, epochs=args.epochs)




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
