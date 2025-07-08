#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np

# mine:
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.metrics
import sklearn.model_selection
import sklearn.neural_network
import sklearn.linear_model
import matplotlib.pyplot as plt
import time
from datetime import datetime
# end mine

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")


class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)


def one_hot_in(data):
    allowed_chars = "abcdefghijklmnopqrstuvwxyz" + "áčďéěíňóřšťúůýž" + " "  # without the " "
    chars_len = len(allowed_chars)
    encoded = np.zeros((len(data), chars_len * len(data[0])))
    for i in range(len(data)):
        for j in range(len(data[0])):
            pos = allowed_chars.find(data[i][j])
            encoded[i][j * chars_len + pos] = 1
    return encoded


def one_hot_out(data):
    allowed_chars = "abcdefghijklmnopqrstuvwxyz" + "áčďéěíňóřšťúůýž" + " "  # without the " "
    chars_len = len(allowed_chars)
    decoded = [[] for _ in range(len(data))]
    for i in range(len(data)):
        for j in range(len(data[0]) // chars_len):
            if 1 in data[i][j * chars_len:(j + 1) * chars_len]:
                pos = np.where(data[i][j * chars_len:(j + 1) * chars_len] == 1)
                decoded[i].append(allowed_chars[pos[0][0]])
            else:
                decoded[i].append(None)
    return decoded


def process_data(input_data, input_target, chunk_len):
    new_data, new_target = [], []
    normal_chars = "abcdefghijklmnopqrstuvwxyz" + "ABCDEFGHIJKLOMNOPQRSTUVWXYZ"
    word_sizes = []  # in chunks

    for i in range(len(input_data)):
        if set(input_data[i]).issubset(set(normal_chars)):
            lower, lower_target = input_data[i].lower(), input_target[i].lower()
            word_sizes.append((len(lower) // chunk_len) + 1)
            padded = list(lower.ljust(((len(lower) // chunk_len) + 1) * chunk_len, " "))
            padded_target = list(lower_target.ljust(((len(lower_target) // chunk_len) + 1) * chunk_len, " "))
            chunks = [padded[j*chunk_len:(j+1)*chunk_len] for j in range(len(padded) // chunk_len)]
            chunks_target = [padded_target[j*chunk_len:(j+1)*chunk_len] for j in range(len(padded_target) // chunk_len)]
            new_data.extend(chunks)
            new_target.extend(chunks_target)

    return new_data, new_target, word_sizes


def finalize_output(original, new_predict, word_sizes):
    new_predict = one_hot_out(new_predict)
    print("lens orig/new_predict/word_sizes:", len(original), len(new_predict), len(word_sizes),
          "sum of word sizes:", np.sum(word_sizes))
    result = ""

    normal_chars = "abcdefghijklmnopqrstuvwxyz" + "ABCDEFGHIJKLOMNOPQRSTUVWXYZ"
    upper_chars = "ABCDEFGHIJKLOMNOPQRSTUVWXYZ" + "áčďéěíňóřšťúůýž".upper()
    predict = []
    k, i = 0, 0
    while i < len(new_predict):
        curr_word = ""
        for j in range(word_sizes[k]):
            piece = ['#' if c is None else c for c in new_predict[i]]
            curr_word += "".join(piece)
            i += 1
        k += 1
        predict.append(curr_word)
    print("predict len (chunks joined):", len(predict))

    # count the interp in original (should be orig-wrd_sizes):
    cnt = 0
    for j in range(len(original)):
        if not set(original[j]).issubset(set(normal_chars)):
            cnt += 1
    print("count the interpunction in original:", cnt, "(should be", len(original) - len(word_sizes), ")")

    k, i = 0, 0
    while i < len(original) and k < len(predict):  # without the second '<'
        while not set(original[i]).issubset(set(normal_chars)):
            result += original[i] + " "
            i += 1
        curr_word = predict[k]
        k += 1

        # swap '#' for original letter
        if '#' in curr_word and len(curr_word.strip()) == len(original[i]):  # without the "=="
            no_tag = ""
            for j in range(len(original[i])):
                if curr_word[j] == '#':
                    no_tag += original[i][j]
                else:
                    no_tag += curr_word[j]
            curr_word = no_tag

        if len(set(original[i]).intersection(upper_chars)) > 0 and len(curr_word.strip()) == len(original[i]):  # without the "=="
            caps = ""
            for j in range(len(original[i])):
                if original[i][j] in upper_chars:
                    caps += curr_word[j].upper()
                else:
                    caps += curr_word[j].lower()
            curr_word = caps

        result += curr_word.strip() + " "
        i += 1

    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"
    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    # hot-fix:
    k, i = 0, 0
    pred = result.split()
    final = ""
    while i < len(original) - 1 and k < len(pred) - 1:
        if len(original[i]) != len(pred[k]):
            if original[i + 1].translate(DIA_TO_NODIA) != pred[k + 1] and len(original[i + 1]) != len(pred[k + 1]):
                print("&&&&")
                final += pred[k] + pred[k + 1] + " "
                k += 2
            else:
                final += pred[k] + " "
                k += 1
        else:
            final += pred[k] + " "
            k += 1
        i += 1
    return final


# hyper-parameters:
chunk_size = 12
test_size = 0.001
mlp_size = 2000
max_iter = 1200


def main(args: argparse.Namespace) -> Optional[str]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # time:
        start_time = time.time()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Started calculation: ", current_time)

        print("hyper params (chunk_size, test_size, mlp_size, max_iter):", chunk_size, test_size, mlp_size, max_iter)

        # model split:
        train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
            train.data.split(), train.target.split(), test_size=test_size, random_state=args.seed, shuffle=False)

        #print("all diacrit were found:", set("áčďéěíňóřšťúůýž").issubset(set("".join(train_target))))

        train_data_p, train_target_p, train_word_sizes = process_data(train_data, train_target, chunk_size)
        test_data_p, test_target_p, test_word_sizes = process_data(test_data, test_target, chunk_size)

        # old one-hot:
        #allowed_chars = "abcdefghijklmnopqrstuvwxyz" + "áčďéěíňóřšťúůýž" + " "  # without the " "
        #features = [list(allowed_chars) for _ in range(chunk_size)]
        #enc = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore', categories=features)
        #enc.fit(train_data_p + train_target_p)
        #train_data_enc = enc.transform(train_data_p)
        #train_target_enc = enc.transform(train_target_p)
        #test_data_enc = enc.transform(test_data_p)
        #test_target_enc = enc.transform(test_target_p)

        train_data_enc = one_hot_in(train_data_p)
        train_target_enc = one_hot_in(train_target_p)
        test_data_enc = one_hot_in(test_data_p)
        test_target_enc = one_hot_in(test_target_p)


        # training:

        mlp = sklearn.neural_network.MLPClassifier(mlp_size, max_iter=max_iter)
        mlp.fit(train_data_enc, train_target_enc)

        # optimize size of mlp:
        mlp._optimizer = None
        for i in range(len(mlp.coefs_)):
            mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
        for i in range(len(mlp.intercepts_)):
            mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)

        model = mlp

        prediction = mlp.predict(test_data_enc)

        accuracy_direct = sklearn.metrics.accuracy_score(prediction, test_target_enc)
        real_approx_acc = (90 * accuracy_direct + 24) / 113
        print("Accuracy on diacritazable: {:.2f}, Approx_real: {:.2f}".format(accuracy_direct, real_approx_acc))

        clean_predict = finalize_output(test_data, prediction, test_word_sizes)
        clean_pred_split = clean_predict.split()

        print("len of test_target:", len(test_target), "len of predict:", len(clean_pred_split))


        # analysis:
        smth = "áčďéěíňóřšťúůýž"
        non_czech = ["čy", "čý", "ďy", "ďý", "ňy", "ňý", "řy", "řý", "šy", "šý", "ťy", "ťý", "žy", "žý"]
        count_all, count_correct = 0, 0
        count_tags, count_u, count_y = 0, 0, 0
        count_long, count_cap = 0, 0
        for i in range(min(len(test_target), len(clean_pred_split))):
            count_all += 1
            if test_target[i] == clean_pred_split[i]:
                count_correct += 1
            else:
                if '#' in clean_pred_split[i]:
                    count_tags += 1
                else:
                    if clean_pred_split[i][0].lower() == 'ů' and test_target[i][0].lower() == 'ú':
                        count_u += 1
                    for bla in non_czech:
                        if bla in clean_pred_split[i].lower():
                            count_y += 1
                if len(clean_pred_split[i]) > 10:
                    count_long += 1
                if test_target[i].upper() == clean_pred_split[i].upper():
                    count_cap += 1

        print("Hand-made accuracy:", count_correct, "/", count_all, "(=", count_correct / count_all, ")")
        print("hashtags:", count_tags, "wrong u/u:", count_u)
        print("(ˇ) and y:", count_y, "long wrong words:", count_long, "wrong capitalization:", count_cap)

        start, size = 0, 100  # 19200
        #print("actual data:  ", test_target[start:start + size])
        #print("predict split:", clean_pred_split[start:start + size])
        #print("actual data end:  ", test_target[-1:-3:-1])
        #print("predict split end:", clean_pred_split[-1:-3:-1])
        #print("clean predict str:", clean_predict[:3])

        # time at finish:
        print("Time of calculation: {:.2f} s ({:.2f} m)".format(
            time.time() - start_time, (time.time() - start_time) / 60))
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Ended calculation: ", current_time)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        test_data = test.data.split()
        test_data_p, _, test_word_sizes = process_data(test_data, test_data, chunk_size)
        test_data_enc = one_hot_in(test_data_p)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        prediction = model.predict(test_data_enc)
        clean_predict = finalize_output(test_data, prediction, test_word_sizes)

        split_orig = test.data.split()
        split_pred = clean_predict.split()

        if len(split_pred) < len(split_orig):
            final_pred = clean_predict
            for i in range(len(split_orig) - len(split_pred)):
                final_pred += " a"
            clean_predict = final_pred
        if len(split_pred) > len(split_orig):
            final_split_pred = split_pred[:len(split_pred) - (len(split_pred) - len(split_orig))]
            clean_predict = " ".join(final_split_pred)


        # TODO: Generate `predictions` with the test set predictions. Specifically,
        # produce a diacritized `str` with exactly the same number of words as `test.data`.
        predictions = clean_predict
        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)