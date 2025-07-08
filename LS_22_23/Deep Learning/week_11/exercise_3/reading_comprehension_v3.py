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
parser.add_argument("--batch_size", default=31, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

# ours:
# hyperparams:
parser.add_argument("--ft_epochs", default=0, type=int, help="Number of FineTune epochs.")
parser.add_argument("--ft_lr", default=0.0001, type=float, help="FineTune learning rate.")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--schedule", default="cos", type=str, help="learning rate schedule")
parser.add_argument("--ft_schedule", default="cos", type=str, help="FineTune learning rate schedule")
parser.add_argument("--take", default=1.0, type=float, help="Take only a fraction of train/dev to speed up debugging.")
parser.add_argument("--take_test", default=1.0, type=float, help="Take only a fraction of test to speed up debugging.")
parser.add_argument("--add_sep", default=1, type=int, help="Add a separator between context and question.")
parser.add_argument("--dropout", default=0.0, type=float, help="Dropout.")


# debugging:
parser.add_argument("--debug_activation", default="sigmoid", type=str, help="for debugging purposes.")
parser.add_argument("--debug_print_trdev_pred", default="no", type=str, help="for debugging purposes.")
parser.add_argument("--debug_offset", default=0, type=int, help="for debugging purposes.")
parser.add_argument("--debug_second_activation", default=0, type=int, help="for debugging purposes.")


class Model(tf.keras.Model):
    def __init__(self, args, backbone):
        inputs = tf.keras.layers.Input(shape=[None], dtype=tf.int32, ragged=True)
        #print("inputs:", inputs)
        hidden = inputs

        row_lengths = hidden.row_lengths()
        mask = tf.sequence_mask(lengths=row_lengths)
        dense = hidden.to_tensor()

        #hidden = backbone(dense, attention_mask=mask)  # is the param called attention_mask?
        hidden = backbone(input_ids=dense, attention_mask=mask)
        #hidden = backbone(input_ids=hidden)

        hidden = hidden.last_hidden_state
        #print("last_hidden_state:", hidden)

        if args.debug_activation == "softmax":
            dense_start_label = tf.keras.layers.Dense(units=1, activation=tf.nn.softmax)
            dense_end_label = tf.keras.layers.Dense(units=1, activation=tf.nn.softmax)
        elif args.debug_activation == "sigmoid":
            dense_start_label = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)
            dense_end_label = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)
        else:
            dense_start_label = tf.keras.layers.Dense(units=1)
            dense_end_label = tf.keras.layers.Dense(units=1)
        start_label = dense_start_label(hidden)
        end_label = dense_end_label(hidden)

        #print("start_char after dense:", start_char)
        #print("end_char after dense:", end_char)
        #print("dense_start_label.count_params()", dense_start_label.count_params())

        dropout_1 = tf.keras.layers.Dropout(rate=args.dropout)
        dropout_2 = tf.keras.layers.Dropout(rate=args.dropout)
        if args.dropout != 0.0:
            start_label = dropout_1(start_label)
            end_label = dropout_2(end_label)

        start_label = tf.squeeze(start_label, axis=-1)
        end_label = tf.squeeze(end_label, axis=-1)
        #print("start_char after squeeze:", start_char)
        #print("end_char after squeeze:", end_char)

        if args.debug_second_activation == 1:
            start_label = tf.nn.softmax(start_label, axis=-1)(start_label)
            end_label = tf.nn.softmax(end_label, axis=-1)(end_label)

        # via mnist_multiple.py
        outputs = {"start_label": start_label, "end_label": end_label}

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
        print("text:", text)
        print("start:", start)
        print("(check:", context[start:start+3], ")")

        tokens = tokenizer.tokenize(context)
        token_ids = tokenizer.encode(context)
        empty_token_ids = tokenizer.encode("")
        print("tokens:", tokens)
        print("token_ids:", token_ids)
        print("empty_token_ids:", empty_token_ids)  # "CLS" ~ 0; "SEP" ~ 2

    def prepare_features(example):
        # concatenate context and question
        context = example["context"]
        qa = example["qas"][0]  # use just the first question (via practicals 11 - 1:33:00)
        question = qa["question"]

        if args.add_sep == 1:  # use the separator token
            context_tokenized = tokenizer.encode(context)
            question_tokenized = tokenizer.encode(question)
            context_and_question = context_tokenized + question_tokenized[1:]
        else:  # concatenate raw text
            context_and_question = tokenizer.encode(context + question)

        #return context_and_question
        return context_and_question, len(example["qas"])

    def prepare_features_multiple(example):
        # for all questions concatenate question with the corresponding context
        context = example["context"]
        qas = example["qas"]  # use all questions for test
        context_and_questions = []

        for qa in qas:
            question = qa["question"]

            if args.add_sep == 1:  # use the separator token
                context_tokenized = tokenizer.encode(context)
                question_tokenized = tokenizer.encode(question)
                context_and_question = context_tokenized + question_tokenized[1:]
            else:  # concatenate raw text
                context_and_question = tokenizer.encode(context + question)
            context_and_questions.append(context_and_question)
        return context_and_questions

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

        # check if we get the same answer_text when we predict:
        """
        cnt = 0
        start_char_span = encoded.token_to_chars(start_token)
        end_char_span = encoded.token_to_chars(end_token)
        if start_char_span is not None and end_char_span is not None:
            start_char = start_char_span.start
            end_char = end_char_span.end
            if start_char < end_char < len(context):
                predicted_answer = context[start_char:end_char]
                if cnt < 10:
                    print("true:", text)
                    print("pred:", predicted_answer)
        """

        label = (start_token, end_token)
        return label

    def create_dataset(name):
        dataset = getattr(rcd, name).paragraphs
        features_array = []
        labels_array = []

        if name != "test":  # test doesnt have labels
            if args.take != 1.0:
                dataset = dataset[:int(np.ceil(len(dataset) * args.take))]
            cnt = 0
            for example in dataset:
                #features = prepare_features(example)
                features, num_q = prepare_features(example)
                cnt += num_q
                features_array.append(features)

                labels = prepare_labels(example)
                labels_array.append(labels)
            print("Number of questions:", cnt)

            features_array = tf.ragged.constant(features_array)
            labels_array = tf.constant(labels_array)

            labels_dict = {"start_label": labels_array[:, 0], "end_label": labels_array[:, 1]}
            dataset = tf.data.Dataset.from_tensor_slices((features_array, labels_dict))
        else:
            if args.take_test != 1.0:
                dataset = dataset[:int(np.ceil(len(dataset) * args.take))]
            for example in dataset:
                features = prepare_features_multiple(example)
                features_array += features
            features_array = tf.ragged.constant(features_array)
            dataset = tf.data.Dataset.from_tensor_slices((features_array, ))  # just "...slices(features_array)" ?
            #dataset = tf.data.Dataset.from_tensor_slices(features_array)

        print("Size:", dataset.cardinality().numpy())
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    print("Creating train dataset.")
    train = create_dataset("train")
    print("Creating dev dataset.")
    dev = create_dataset("dev")
    print("Creating test dataset.")
    test = create_dataset("test")

    print("Datasets prepared. (phew)")

    # =================================================================================================================
    # MODEL CREATION:
    # =================================================================================================================

    model = Model(args=args, backbone=robeczech)

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
        offset = args.debug_offset
        test = rcd.test.paragraphs
        answers = []
        counter = 0

        for i in range(len(test)):
            context = test[i]["context"]
            encoded = tokenizer(context)
            num_of_questions = len(test[i]["qas"])

            for j in range(num_of_questions):
                start_token = tf.argmax(predictions["start_label"][counter])
                end_token = tf.argmax(predictions["end_label"][counter])

                start_char_span = encoded.token_to_chars(start_token + offset)
                end_char_span = encoded.token_to_chars(end_token + offset)

                answer = ""
                if start_char_span is not None and end_char_span is not None:
                    start_char = start_char_span.start
                    end_char = end_char_span.end
                    if start_char < end_char < len(context):
                        answer = context[start_char:end_char]
                answers.append(answer)
                counter += 1
        return answers

    def print_debug_predictions(predictions, dataset):
        print("predictions['start_label'][0]:", predictions["start_label"][0])
        offset = args.debug_offset
        if args.take != 1.0:
            dataset = dataset[:int(np.ceil(len(dataset) * args.take))]

        for i in range(len(dataset)):
            context = dataset[i]["context"]
            encoded = tokenizer(context)

            true_answer = dataset[i]["qas"][0]["answers"][0]["text"]
            true_start = dataset[i]["qas"][0]["answers"][0]["start"]

            true_answer_length = len(true_answer)

            true_start_token = encoded.char_to_token(true_start)
            true_end_token = encoded.char_to_token(true_start + true_answer_length - 1)

            start_token = tf.argmax(predictions["start_label"][i])
            end_token = tf.argmax(predictions["end_label"][i])

            start_char_span = encoded.token_to_chars(start_token + offset)
            end_char_span = encoded.token_to_chars(end_token + offset)

            predicted_answer = ""
            if start_char_span is not None and end_char_span is not None:
                start_char = start_char_span.start
                end_char = end_char_span.end
                if start_char < end_char < len(context):
                    predicted_answer = context[start_char:end_char]

            if i < 30:
                print("True start token:", true_start_token)
                print("Pred start token:", start_token.numpy())
                print("True end token:  ", true_end_token)
                print("Pred end token:  ", end_token.numpy())
                print("True answer:     ", true_answer)
                print("Pred answer:     ", predicted_answer)

    if args.debug_print_trdev_pred == "train":
        print("Showing true answers vs predicted answers on the Train dataset:")
        debug_pred = model.predict(train)
        print_debug_predictions(predictions=debug_pred, dataset=rcd.train.paragraphs)
    elif args.debug_print_trdev_pred == "dev":
        print("Showing true answers vs predicted answers on the Dev dataset:")
        debug_pred = model.predict(dev)
        print_debug_predictions(predictions=debug_pred, dataset=rcd.dev.paragraphs)

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
