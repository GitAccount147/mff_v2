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
import sklearn.model_selection
import sklearn.neural_network
import time
from datetime import datetime
from scipy.sparse import csr_array
from scipy.sparse import coo_array
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


class Dictionary:
    def __init__(self,
                 name="fiction-dictionary.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dictionary to `variants`
        self.variants = {}
        with open(name, "r", encoding="utf-8-sig") as dictionary_file:
            for line in dictionary_file:
                nodia_word, *variants = line.rstrip("\n").split()
                self.variants[nodia_word] = variants


def one_hot_in(data):
    chars = ["abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "0123456789", " !,.?", """()"'-:;"""]
    allowed_chars = """ !"'(),-.0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"""
    chars_len = len(allowed_chars)

    #encoded = np.zeros((len(data), chars_len * len(data[0])), dtype=np.float16)
    #encoded = np.zeros((len(data), chars_len * len(data[0])))
    #encoded = np.zeros((len(data), chars_len * len(data[0])), dtype=np.dtype('u1'))
    #encoded = csr_array((len(data), chars_len * len(data[0])), dtype=np.float64)
    #encoded = coo_array((len(data), chars_len * len(data[0])), dtype=np.float64)

    row_py = []
    col_py = []
    data_py = []

    #print("tady", encoded.shape)
    #print(type(encoded[0][0]))
    for i in range(len(data)):
        for j in range(len(data[0])):
            pos = allowed_chars.find(data[i][j])
            if pos != -1:
                #encoded[i][j * chars_len + pos] = 1
                row_py.append(i)
                col_py.append(j * chars_len + pos)
                data_py.append(1)
    row = np.array(row_py)
    col = np.array(col_py)
    array_data = np.array(data_py)
    encoded = coo_array((array_data, (row, col)), shape=(len(data), chars_len * len(data[0])))
    return encoded


def one_hot_out(data):
    chars = ["abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "0123456789", " !,.?", """()"'-:;"""]
    allowed_chars = """ !"'(),-.0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"""
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


def target_transform(target):
    result = []
    result_no_undia = []
    letters_dia = "áčďéěíňóřšťúůýž"
    letters_no_dia = "acdeinorstuyz"
    #letters = "abcdefghijklmnopqrstuvwxyz"
    signature = [1, 2, 2, 1, 2, 1, 2, 1, 2, 2, 2, 1, 3, 1, 2]
    for i in range(len(target)):
        if target[i].lower() in letters_dia:
            result.append(signature[letters_dia.index(target[i].lower())])
        else:
            result.append(0)
        if target[i].lower() in letters_dia:
            result_no_undia.append(signature[letters_dia.index(target[i].lower())])
        elif target[i].lower() in letters_no_dia:
            result_no_undia.append(0)
    return result, result_no_undia


def apply_signature(data, signature):
    result = ""
    letters_nodia = "acdeinorstuyz"
    table = {"a": ["á", None, None], "c": [None, "č", None], "d": [None, "ď", None], "e": ["é", "ě", None],
             "i": ["í", None, None], "n": [None, "ň", None], "o": ["ó", None, None], "r": [None, "ř", None],
             "s": [None, "š", None], "t": [None, "ť", None], "u": ["ú", None, "ů"], "y": ["ý", None, None],
             "z": [None, "ž", None]}
    for i in range(len(data)):
        if signature[i] != 0 and data[i].lower() in letters_nodia:
            if table[data[i].lower()][signature[i] - 1] is not None:
                if data[i].lower() == data[i]:
                    result += table[data[i]][signature[i] - 1]
                else:
                    result += table[data[i].lower()][signature[i] - 1].upper()
            else:
                result += data[i]
        else:
            result += data[i]
    return result


def data_transform(data):
    result = []
    dia = "acdeinorstuyz"
    undia_chars = ""
    undia_pos = []
    res_no_pad = ""
    padded_data = " " * front_feature_size + data + " " * back_feature_size
    chunk_length = front_feature_size + back_feature_size + 1
    for i in range(len(data)):
        if data[i].lower() in dia:
            result.append(padded_data[i:i + chunk_length])
            res_no_pad += data[i]
        else:
            undia_chars += data[i]
            undia_pos.append(i)
    return result, undia_chars, undia_pos, res_no_pad


def glue_back(data, undia_chars, undia_pos):
    result = ""
    tot_len = len(data) + len(undia_chars)
    j, k = 0, 0
    for i in range(tot_len):
        if i == undia_pos[j]:
            result += undia_chars[j]
            j += 1
        else:
            result += data[k]
            k += 1
    return result


def un_dia(word):
    letters_nodia = "acdeeinorstuuyz"
    letters_dia = "áčďéěíňóřšťúůýž"
    return word.translate(str.maketrans(letters_dia + letters_dia.upper(), letters_nodia + letters_nodia.upper()))


def correct_with_dict(prediction, dictionary):
    result = []
    predicted_words = prediction.split()
    for i in range(len(predicted_words)):
        word_no_dia = un_dia(predicted_words[i])
        if word_no_dia in dictionary:
            words = dictionary[word_no_dia]
            if len(words) == 1:
                result.append(words[0])
            else:
                mistakes = [0 for _ in range(len(words))]
                for j in range(len(words)):
                    for k in range(len(words[j])):
                        if words[j][k] != predicted_words[i][k]:
                            mistakes[j] += 1
                result.append(words[np.argmin(mistakes)])
        else:
            result.append(predicted_words[i])
    return " ".join(result)


# hyper-params:
test_size = 0.01
front_feature_size = 6
back_feature_size = 6
mlp_size = 50
max_iter = 200


def main(args: argparse.Namespace) -> Optional[str]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        dict = Dictionary()

        # time:
        start_time = time.time()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Started calculation: ", current_time)
        print("hyper params (chunk_size, test_size, mlp_size, max_iter):",
              front_feature_size + back_feature_size + 1, "= (", front_feature_size, "+", back_feature_size, "+ 1 )",
              test_size, mlp_size, max_iter)


        # analyse the dictionary:

        word = "bilo"
        #print(word in dict.variants)
        """
        sss, kkk = 0, 0
        lens = [0 for _ in range(10)]
        for key in dict.variants:
            sss += len(dict.variants[key])
            kkk += 1
            lens[len(dict.variants[key])] += 1
            if len(dict.variants[key]) >= 7:
                kkk += 0
                #print(dict.variants[key])
        """
        #print(sss, kkk, lens)
        #print(len(dict.variants))


        # old model split:
        #train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        #    train.data, train.target, test_size=test_size, random_state=args.seed, shuffle=False)

        # model split - split data without splitting words:

        split_mark = round((1 - test_size) * len(train.data))
        while train.data[split_mark] != ' ':
            split_mark += 1
        train_data, test_data = train.data[:split_mark], train.data[split_mark:]
        train_target, test_target = train.target[:split_mark], train.target[split_mark:]


        # testing the new functions:

        #print(train_data)
        #print(test_data[:20])
        #print("".join(sorted(set(train.data))))
        #print(""" !"'(),-.01246789:;?ABCDEFGHIJKLMNOPRSTUVWXYZabcdefghijklmnoprstuvwxyz"""[1])
        #print(train_target[:50], target_transform(train_target[:50]))
        #print(apply_signature("avesuu", [0, 2, 1, 2, 2, 3]))
        #print(train_target == apply_signature(train_data, target_transform(train_target)))
        #print(data_transform("sjdvsjkdvbsjkvbs"))
        #print(un_dia("ěíwefňósdvřšťú"))
        #print(correct_with_dict("býl bilo Alžbetínů", dict.variants))



        train_data_trans, _, _, _ = data_transform(train_data)
        trans_time = time.time()
        train_data_enc = one_hot_in(train_data_trans)
        enc_time = time.time()
        _, train_target_enc = target_transform(train_target)
        enc_target_time = time.time()
        #print("size of train_data_enc (MB):", train_data_enc.nbytes / 1048576)
        print("size of train_data_enc/train_data_trans/train_target_enc (MB): {:.2f}/{:.2f}/{:.2f}".format(
            sys.getsizeof(train_data_enc) / 1048576, sys.getsizeof(train_data_trans) / 1048576,
            sys.getsizeof(train_target_enc) / 1048576))
        # (11 feat, test_size 0.1) 3301766336 bytes ~ 3GB
        print("time of trans/enc/target_enc:",
               trans_time - start_time, enc_time - trans_time, enc_target_time - enc_time)

        test_data_trans, unused_test, unused_test_pos, test_no_pad = data_transform(test_data)
        print("size of test_data_trans/unused_test/unused_test_pos/test_no_pad (MB): {:.2f}/{:.2f}/{:.2f}/{:.2f}".format(
              sys.getsizeof(test_data_trans) / 1048576, sys.getsizeof(unused_test) / 1048576,
              sys.getsizeof(unused_test_pos) / 1048576, sys.getsizeof(test_no_pad) / 1048576))
        test_data_enc = one_hot_in(test_data_trans)

        mlp = sklearn.neural_network.MLPClassifier(mlp_size, max_iter=max_iter)
        mlp.fit(train_data_enc, train_target_enc)

        # optimize size of mlp:
        mlp._optimizer = None
        for i in range(len(mlp.coefs_)):
            mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
        for i in range(len(mlp.intercepts_)):
            mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)


        test_and_fit_time = time.time()
        predict = mlp.predict(test_data_enc)
        predict_time = time.time()
        #print(len(test_no_pad), len(predict))
        predict_clean = apply_signature(test_no_pad, predict)
        signature_time = time.time()
        predict_glued = glue_back(predict_clean, unused_test, unused_test_pos)
        predict_dict = correct_with_dict(predict_glued, dict.variants)
        dict_time = time.time()
        print("time of (test+fit)/predict/signature/(dict+glue):",
              test_and_fit_time - enc_target_time, predict_time - test_and_fit_time, signature_time - predict_time,
              dict_time - signature_time)
        print("size of predict/predict_clean/predict_glued/predict_dict (MB): {:.2f}/{:.2f}/{:.2f}/{:.2f}".format(
              sys.getsizeof(predict) / 1048576, sys.getsizeof(predict_clean) / 1048576,
              sys.getsizeof(predict_glued) / 1048576, sys.getsizeof(predict_dict) / 1048576))


        # accuracy:
        split_predict = predict_dict.split()
        split_target = test_target.split()
        count_correct, count_all = 0, 0
        for i in range(len(split_predict)):
            if split_predict[i] == split_target[i]:
                count_correct += 1
            count_all += 1
        print("Hand counted accuracy:", count_correct / count_all)

        # time at finish:
        print("Time of calculation: {:.2f} s ({:.2f} m)".format(
            time.time() - start_time, (time.time() - start_time) / 60))
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Ended calculation: ", current_time)




        # TODO: Train a model on the given dataset and store it in `model`.
        model = mlp

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)
        dict_dia = Dictionary()

        test_data = test.data

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        test_data_trans, unused_test, unused_test_pos, test_no_pad = data_transform(test_data)
        test_data_enc = one_hot_in(test_data_trans)

        predict = model.predict(test_data_enc)
        predict_clean = apply_signature(test_no_pad, predict)
        predict_glued = glue_back(predict_clean, unused_test, unused_test_pos)
        predict_dict = correct_with_dict(predict_glued, dict_dia.variants)
        #predict_dict = predict_clean

        # TODO: Generate `predictions` with the test set predictions. Specifically,
        # produce a diacritized `str` with exactly the same number of words as `test.data`.
        predictions = predict_dict

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)