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


def main(args: argparse.Namespace) -> Optional[str]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        #train_data, train_target = train.data.split(), train.target.split()

        same = 0
        diff = 0
        test1, test2 = train.data.split(), train.target.split()
        for i in range(len(test1)):
            if test1[i] == test2[i]:
                same += 1
            else:
                diff += 1
        print("words without diacr:", same, "words with diacr:", diff)

        # split to train and test:
        test_size = 0.5
        train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
            train.data.split(), train.target.split(), test_size=test_size, random_state=args.seed)

        # longest words:
        # print(train_data[:10])
        # print(train_target[:10])
        lens = [0 for _ in range(25)]
        LETTERS_NODIA = "acdeeinorstuuyz"
        undia_chars = LETTERS_NODIA + LETTERS_NODIA.upper()
        undiac_single = 0
        undiac_all = 0
        for word in train_data + test_data:
            lens[len(word)] += 1
            if len(word) == 1 and word[0] not in undia_chars:
                undiac_single += 1
            if len(word) >= 15:
                _ = 0
                # print(word)
            if len(set(word).intersection(undia_chars)) == 0:
                undiac_all += 1
        # np.set_printoptions(suppress=True)
        print("single char words, that cannot have diacrit:", undiac_single)
        print("all words, that cannot have diacrit:", undiac_all)
        print("total num of words:", len(train_data + test_data))
        print(lens)
        plt.plot(lens, 'o')
        plt.show()


        # pad ' ' to max length
        for i in range(len(train_data)):
            train_data[i] = list(train_data[i].ljust(20, ' '))
            train_target[i] = list(train_target[i].ljust(20, ' '))
        for i in range(len(test_data)):
            test_data[i] = list(test_data[i].ljust(20, ' '))
            test_target[i] = list(test_target[i].ljust(20, ' '))
        print("train_data and test_data length:", len(train_data), len(test_data))
        #print(train_data[:5])




        # alphabet symbols:
        print(set("aaavvvvddd") == set("avd"), set("aaaavvd") == set("aav"), set("aaavvd").issubset(set("aavff")))
        alph_str = "abcdefghijklmnopqrstuvwxyz"
        alph = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
                "v", "w", "x", "y", "z"]
        alph_str_dia = alph_str + "áčďéěíňóřšťúůýž"
        alph_dia = alph + ["á", "č", "ď", "é", "ě", "í", "ň", "ó", "ř", "š", "ť", "ú", "ů", "ý", "ž"]
        alph_str_all = alph_str_dia + alph_str_dia.upper() + ' ' + '.,?!'
        alph_final = list(alph_str_all)
        #print(alph_final)
        #print(alph_str_all)
        #print(alph_str_dia, alph_dia)

        # one-hot encode:
        """
        enc2 = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')
        x2 = [['M', "g", ' '], ['F', "c", ' '], ['F', "d", ' ']]
        enc2.fit(x2)
        print(enc2.transform([['F', "g", ' '], ['M', "a", ' ']]).toarray())
        for i in range(5):
            print(len(train_data[i]))

        ttt = train_data[:2]
        print(ttt)
        """

        chars = sorted(list(set(train.data + train.target)))
        print("number of features (per char ~> * 20):", len(chars))
        #print(chars)
        chars20 = [chars for _ in range(20)]
        #print(chars20)
        enc = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore', categories=chars20)
        # enc.fit([['M', 'e', 'l', 'a', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        #         ['n', 'a', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']])
        enc.fit(train_data)
        #print(enc.categories_)
        train_data = enc.transform(train_data)
        test_data = enc.transform(test_data)
        #print(train_data, test_data)
        train_target = enc.transform(train_target)
        test_target = enc.transform(test_target)
        #print((one_hot_test[0].toarray())[0][:20])


        # training:
        scaler = sklearn.preprocessing.StandardScaler()

        #mlp = sklearn.neural_network.MLPRegressor()
        mlp = sklearn.neural_network.MLPClassifier(50)
        #mlp.fit(train_data, train_target)
        #pred = mlp.predict(test_data)

        sgd = sklearn.linear_model.SGDClassifier(verbose=10)
        #sgd.fit(train_data, train_target)
        #pred = sgd.predict(test_data)

        #pipe = sklearn.pipeline.Pipeline([('scaler', scaler), ('sgd', sgd)])
        #pipe = sklearn.pipeline.Pipeline([('scaler', scaler), ('mlp', mlp)])
        pipe = sklearn.pipeline.Pipeline([('mlp', mlp)])
        pipe.fit(train_data, train_target)
        print("done with fitting")
        pred = pipe.predict(test_data)

        accuracy = sklearn.metrics.accuracy_score(pred, test_target)
        print(accuracy)


        # TODO: Train a model on the given dataset and store it in `model`.
        model = ...

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions. Specifically,
        # produce a diacritized `str` with exactly the same number of words as `test.data`.
        predictions = ...

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)