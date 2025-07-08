#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
import transformers

from reading_comprehension_dataset import ReadingComprehensionDataset

# "Bez motivace nejsou koláče. Nemáme koláče."
# 3d76595a-e687-11e9-9ce9-00505601122b
# 594215cf-e687-11e9-9ce9-00505601122b

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=37, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

# ours:
parser.add_argument("--ft_epochs", default=0, type=int, help="Number of FineTune epochs.")
parser.add_argument("--ft_lr", default=0.0001, type=float, help="FineTune learning rate.")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--schedule", default="cos", type=str, help="learning rate schedule")
parser.add_argument("--ft_schedule", default="cos", type=str, help="FineTune learning rate schedule")
parser.add_argument("--take", default=0.01, type=float, help="Take only a fraction of datasets to speed up debugging.")
parser.add_argument("--add_sep", default=1, type=int, help="Add a separator between context and question.")
#parser.add_argument("--dropout", default=0.0, type=float, help="Dropout.")


class Model(tf.keras.Model):
    def __init__(self, backbone):
        inputs = tf.keras.layers.Input(shape=[None], dtype=tf.int32, ragged=True)
        hidden = inputs

        row_lengths = hidden.row_lengths()
        mask = tf.sequence_mask(lengths=row_lengths)
        dense = hidden.to_tensor()

        hidden = backbone(dense, attention_mask=mask)

        hidden = hidden.last_hidden_state
        #print("last_hidden_state:", hidden)

        hidden = tf.RaggedTensor.from_tensor(tensor=hidden, lengths=row_lengths)

        #start_char = tf.keras.layers.Dense(units=1, activation=tf.nn.softmax)(hidden)
        #end_char = tf.keras.layers.Dense(units=1, activation=tf.nn.softmax)(hidden)
        start_char = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)(hidden)
        end_char = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)(hidden)
        #start_char = tf.keras.layers.Dense(units=1)(hidden)
        #end_char = tf.keras.layers.Dense(units=1)(hidden)
        #start_char = hidden
        #end_char = hidden
        #print("start_char after dense:", start_char)
        #print("end_char after dense:", end_char)

        start_char = start_char[0, :]
        end_char = end_char[0, :]

        #start_char = tf.squeeze(start_char)
        #end_char = tf.squeeze(end_char)
        #print("start_char after squeeze:", start_char)
        #print("end_char after squeeze:", end_char)

        #start_char = tf.argmax(start_char)
        #end_char = tf.argmax(end_char)
        #print("start_char after argmax:", start_char)
        #print("end_char after argmax:", end_char)

        #start_char = tf.cast(start_char, dtype=tf.int32)
        #end_char = tf.cast(end_char, dtype=tf.int32)
        #print("start_char after cast:", start_char)
        #print("end_char after cast:", end_char)

        #start_char = hidden[:, 0]
        #end_char = hidden[:, 0]

        # via mnist_multiple.py
        outputs = {"start_label": start_char, "end_label": end_char}

        super().__init__(inputs=inputs, outputs=outputs)

        self.output_names = sorted(outputs.keys())

        self.backbone = backbone


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

    # Load the pre-trained RobeCzech model
    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/robeczech-base")
    robeczech = transformers.TFAutoModel.from_pretrained("ufal/robeczech-base")

    #object_methods = [method_name for method_name in dir(robeczech) if callable(getattr(robeczech, method_name))]
    #print(object_methods)
    #print(dir(robeczech))
    #print(len(robeczech.trainable_variables))
    #print(robeczech.trainable_weights)
    #for weight in robeczech.trainable_weights:
    #    print(tf.shape(weight))

    # Load the data
    rcd = ReadingComprehensionDataset()

    # TODO: Create the model and train it
    #model = ...

    # =================================================================================================================
    # DATA PREPARATION:
    # =================================================================================================================

    print_example = False
    if print_example:
        train = rcd.train.paragraphs
        #print(train[0])

        paragraph_num = 0
        context = train[paragraph_num]["context"]
        qa = train[paragraph_num]["qas"][0]
        question = qa["question"]
        text = qa["answers"][0]["text"]
        start = qa["answers"][0]["start"]  # starting char position
        print("context:", context)
        print("question:", question)
        print("text", text)
        print("start:", start)
        print("(check:", context[start:start+3], ")")

        tokens = tokenizer.tokenize(context)
        token_ids = tokenizer.encode(context)
        empty_token_ids = tokenizer.encode("")
        print("tokens:", tokens)
        print("token_ids:", token_ids)
        print("empty_token_ids:", empty_token_ids)  # "CLS" ~ 0; "SEP" ~ 2

    add_sep = args.add_sep

    def prepare_features(example):
        context = example["context"]
        qa = example["qas"][0]  # use just the first question (via practicals 11 - 1:33:00)
        question = qa["question"]

        if add_sep == 1:
            context_tokenized = tokenizer.encode(context)
            question_tokenized = tokenizer.encode(question)
            question_tokenized[0] = 2  # "SEP"
            context_and_question = context_tokenized + question_tokenized
        else:
            context_and_question = tokenizer.encode(context + question)

        return context_and_question

    def prepare_labels(example):
        qa = example["qas"][0]  # use just the first question (via practicals 11 - 1:33:00)
        text = qa["answers"][0]["text"]  # use just the first answer (via practicals 11 - 1:33:00)
        start = qa["answers"][0]["start"]

        # via bert_example.py
        context = example["context"]
        encoded = tokenizer(context)
        answer_length = len(text)

        start_token = encoded.char_to_token(start)
        end_token = encoded.char_to_token(start + answer_length - 1)  # (start+answer_len) sometimes lands on non-letter

        #print(start, start_token, start + answer_length, end_token, len(context))

        #label = (start, start + answer_length)
        label = (start_token, end_token)
        return label

    def prepare_dataset(example):
        context = example["context"]
        qa = example["qas"][0]  # use just the first question (via practicals 11 - 1:33:00)
        question = qa["question"]
        text = qa["answers"][0]["text"]  # use just the first answer (via practicals 11 - 1:33:00)
        start = qa["answers"][0]["start"]

        if add_sep:
            context_tokenized = tokenizer.encode(context)
            question_tokenized = tokenizer.encode(question)
            question_tokenized[0] = 2  # "SEP"
            context_and_question = context_tokenized + question_tokenized
        else:
            context_and_question = tokenizer.encode(context + question)

        answer_length = len(text)
        label = (start, start + answer_length)
        return context_and_question, label

    def prepare_test_dataset(example):
        context = example["context"]
        qa = example["qas"][0]  # use just the first question (via practicals 11 - 1:33:00)
        question = qa["question"]

        if add_sep:
            context_tokenized = tokenizer.encode(context)
            question_tokenized = tokenizer.encode(question)
            question_tokenized[0] = 2  # "SEP"
            context_and_question = context_tokenized + question_tokenized
        else:
            context_and_question = tokenizer.encode(context + question)

        return context_and_question

    def create_dataset(name):
        dataset = getattr(rcd, name).paragraphs
        features_array = []
        labels_array = []
        if name != "test":  # test doesnt have labels
            #dataset = dataset.map(prepare_dataset)
            if args.take != 0.0:
                dataset = dataset[:int(np.ceil(len(dataset) * args.take))]
            for example in dataset:
                features = prepare_features(example)
                features_array.append(features)

                labels = prepare_labels(example)
                #print(labels[0], labels[1], type(labels[0]), type(labels[1]))
                #labels = {"start_label": labels[0], "end_label": labels[1]}
                labels_array.append(labels)
            #print(len(features_array), len(labels_array))
            features_array = tf.ragged.constant(features_array)
            labels_array = tf.constant(labels_array)

            # maybe useless:
            #features_array = tf.cast(features_array, dtype=tf.int32)
            #labels_array = tf.cast(labels_array, dtype=tf.int32)

            #print("features_array.dtype:", features_array.dtype)
            #print("labels_array.dtype:", labels_array.dtype)

            labels_dict = {"start_label": labels_array[:, 0], "end_label": labels_array[:, 1]}
            #dataset = tf.data.Dataset.from_tensor_slices((features_array, labels_array))
            dataset = tf.data.Dataset.from_tensor_slices((features_array, labels_dict))
        else:
            #dataset = dataset.map(prepare_test_dataset)
            for example in dataset:
                features = prepare_features(example)
                features_array.append(features)
            features_array = tf.ragged.constant(features_array)
            #features_array = tf.cast(features_array, dtype=tf.int32)
            dataset = tf.data.Dataset.from_tensor_slices(features_array)

        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    train, dev = create_dataset("train"), create_dataset("dev")
    test = create_dataset("test")

    print("Datasets prepared. (phew)")

    # =================================================================================================================
    # MODEL CREATION:
    # =================================================================================================================

    model = Model(backbone=robeczech)

    if args.schedule == "cos":
        lr = tf.optimizers.schedules.CosineDecay(args.lr, args.epochs * len(train))
    else:
        lr = args.lr

    if args.ft_schedule == "cos":
        ft_lr = tf.optimizers.schedules.CosineDecay(args.ft_lr, args.ft_epochs * len(train))
    else:
        ft_lr = args.ft_lr

    # TRAINING:
    model.backbone.trainable = False
    model.compile(optimizer=tf.optimizers.Adam(jit_compile=False, learning_rate=lr),
                  loss={"start_label": tf.keras.losses.SparseCategoricalCrossentropy(),
                        "end_label": tf.keras.losses.SparseCategoricalCrossentropy()
                        },
                  #loss={"start_label": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  #      "end_label": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                  #      },
                  metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],  # use evaluate from dataset?
                  )
    model.fit(train, epochs=args.epochs, validation_data=dev)

    # FINE-TUNING:
    if args.ft_epochs != 0:
        model.backbone.trainable = True
        model.compile(optimizer=tf.optimizers.Adam(jit_compile=False, learning_rate=ft_lr),
                      loss={"start_label": tf.keras.losses.SparseCategoricalCrossentropy(),
                            "end_label": tf.keras.losses.SparseCategoricalCrossentropy()
                            },
                      metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
                      )
        model.fit(train, epochs=args.ft_epochs, validation_data=dev)

    # =================================================================================================================
    # PREDICTION:
    # =================================================================================================================

    def predicted_tokens_to_words(predictions):
        print("predictions", predictions)
        test = rcd.test.paragraphs
        print("test[0]:", test[0])
        answers = []
        """
        for i in range(len(test)):
            context = test[i]["context"]
            #start_token, end_token = predictions[i]["start_label"], predictions[i]["end_label"]
            start_token, end_token = predictions[i][0], predictions[i][1]
            encoded = tokenizer(context)
            start_char = encoded.token_to_chars(start_token).start
            end_char = encoded.token_to_chars(end_token).end
            if start_char < end_char < len(context):
                answer = context[start_char:end_char]
            else:
                answer = ""
            answers.append(answer)
        """
        for i in range(len(test)):
            batch = test[i]
            for j in range(len(batch)):
                context = batch[j]["context"]
                #start_token, end_token = predictions[i]["start_label"], predictions[i]["end_label"]
                start_token, end_token = predictions[i][j][0], predictions[i][j][1]
                encoded = tokenizer(context)
                start_char = encoded.token_to_chars(start_token).start
                end_char = encoded.token_to_chars(end_token).end
                if start_char < end_char < len(context):
                    answer = context[start_char:end_char]
                else:
                    answer = ""
                answers.append(answer)
        return answers

    token_predictions = model.predict(test)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "reading_comprehension.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the answers as strings, one per line.
        #predictions = ...
        #predictions = model.predict(test)
        predictions = predicted_tokens_to_words(token_predictions)

        for answer in predictions:
            print(answer, file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
