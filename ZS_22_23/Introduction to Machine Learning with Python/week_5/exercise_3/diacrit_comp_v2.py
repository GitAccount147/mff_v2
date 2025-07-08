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


def process_data(input_data, input_target, test_size, seed, chunk_len):
    input_data, input_target = input_data.split(), input_target.split()
    #print(input_data, input_target)

    char_dia = "acdeinorstuyz"  # without duplicates

    new_data, new_target = [], []
    #chunk_len = 5
    normal_chars = "abcdefghijklmnopqrstuvwxyz" + "ABCDEFGHIJKLOMNOPQRSTUVWXYZ"

    for i in range(len(input_data)):
        #if len(input_data[i]) > 1 or input_data[i][0] in normal_chars:  # in char_dia
            #print("non-interp")
        if set(input_data[i]).issubset(set(normal_chars)):
            #print("non-weird")
            lower, lower_target = input_data[i].lower(), input_target[i].lower()
            padded = list(lower.ljust(((len(lower) // chunk_len) + 1) * chunk_len, ' '))
            #print(padded)
            padded_target = list(lower_target.ljust(((len(lower_target) // chunk_len) + 1) * chunk_len, ' '))
            chunks = [padded[j*chunk_len:(j+1)*chunk_len] for j in range(len(padded) // chunk_len)]
            #print(chunks)
            chunks_target = [padded_target[j*chunk_len:(j+1)*chunk_len] for j in range(len(padded_target) // chunk_len)]
            new_data.extend(chunks)
            new_target.extend(chunks_target)

    #print(new_data, new_target)

    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        new_data, new_target, test_size=test_size, random_state=seed)

    return train_data, test_data, train_target, test_target


def finalize_output(original, new_predict, enc):
    #original.split()
    new_predict = enc.inverse_transform(new_predict)
    print(new_predict)
    result = ""
    i, j = 0, 0
    normal_chars = "abcdefghijklmnopqrstuvwxyz" + "ABCDEFGHIJKLOMNOPQRSTUVWXYZ"
    while j < len(original):
        curr_word = "".join(new_predict[i])
        print(curr_word)
        i += 1
        while curr_word[-1] != ' ':
            curr_word += "".join(new_predict[i])
            print(curr_word)
            i += 1

        while not set(original[j]).issubset(set(normal_chars)):
            result += original[j] + " "
            j += 1
        #while curr_word != original[j]:

        result += curr_word + " "
        j += 1

    return result



def main(args: argparse.Namespace) -> Optional[str]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        #print(train.data.split()[:100])

        # time:
        start_time = time.time()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Started calculation: ", current_time)

        # hyper-parameters:
        chunk_size = 7
        test_size = 0.5
        folds = 5

        train_data, test_data, train_target, test_target = process_data(
            train.data, train.target, test_size, args.seed, chunk_size)
        #print(train_data, test_data)
        #print(train_target, test_target)

        # one-hot:
        allowed_chars = "abcdefghijklmnopqrstuvwxyz" + "áčďéěíňóřšťúůýž" #+ " " # without the " "
        features = [list(allowed_chars) for _ in range(chunk_size)]
        enc = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore', categories=features)
        enc.fit(train_data + train_target)
        train_data_e = enc.transform(train_data)
        test_data_e = enc.transform(test_data)
        train_target_e = enc.transform(train_target)
        test_target_e = enc.transform(test_target)
        #print(train_data)


        # training:

        # ranges and params:
        layer_sizes = [[(100,)], [(200,)], [(50,)], [(400,), (800,)], [(800,)], [(50,), (100,), (200,)], [(10, 10), (50, 50), (100, 100)],
                       [(400,), (200, 200), (400, 200, 100), (400, 200, 100, 50)], [(400, 400), (800,), (800, 800)]]
        max_iter = [[200], [200, 400], [800]]

        parameters = [{'mlp__hidden_layer_sizes': layer_sizes[4], 'mlp__max_iter': max_iter[2]}]

        scaler = sklearn.preprocessing.StandardScaler()
        sgd = sklearn.linear_model.SGDClassifier()
        mlp = sklearn.neural_network.MLPClassifier()

        pipe = sklearn.pipeline.Pipeline([('mlp', mlp)])
        #pipe = sklearn.pipeline.Pipeline([('sgd', sgd)])
        #pipe = sklearn.pipeline.Pipeline([('scaler', scaler), ('mlp', mlp)])
        #pipe.fit(train_data_e, train_target_e)
        #prediction = pipe.predict(test_data_e)

        #print(prediction[:20])
        #print("test data:", test_data[:10])
        #print("test target:", test_target[:10])
        #print(finalize_output(train.data.split[:10], prediction[:10], enc))

        skf = sklearn.model_selection.StratifiedKFold(n_splits=folds)
        # model = sklearn.model_selection.GridSearchCV(pipe, param_grid=parameters, cv=skf, refit=True, verbose=4)
        # model = sklearn.model_selection.GridSearchCV(pipe, param_grid=parameters, cv=skf, refit=True)
        model = sklearn.model_selection.GridSearchCV(pipe, param_grid=parameters, refit=True, verbose=4)
        model.fit(train_data_e.toarray(), train_target_e.toarray())
        real_approx_acc = (90 * model.best_score_ + 24) / 113
        print("best score:", model.best_score_ * 100, "%, while using:", model.best_params_)
        print("approx real best score:", real_approx_acc * 100)

        #accuracy = sklearn.metrics.accuracy_score(prediction, test_target)
        #real_approx_acc = (90 * accuracy + 24) / 113
        #print("Accuracy on diacritazable: {:.2f}, Approx_real: {:.2f}".format(accuracy, real_approx_acc))

        # time at finish:
        print("Time of calculation: {:.2f} s ({:.2f} m)".format(
            time.time() - start_time, (time.time() - start_time) / 60))
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Ended calculation: ", current_time)


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