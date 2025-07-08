#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt

# moje:
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

import sklearn.linear_model
import sklearn.metrics
# konec moje

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")


class Dataset:
    """Rental Dataset.
    The dataset instances consist of the following 12 features:
    - season (1: winter, 2: spring, 3: summer, 4: autumn)
    - year (0: 2011, 1: 2012)
    - month (1-12)
    - hour (0-23)
    - holiday (binary indicator)
    - day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - working day (binary indicator; a day is neither weekend nor holiday)
    - weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - relative humidity (0-1 range)
    - windspeed (normalized to 0-1 range)
    The target variable is the number of rented bikes in the given hour.
    """
    def __init__(self,
                 name="rental_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()


        #print(train.data, train.target)
        #print(train.data.shape, train.target.shape)
        data, target = train.data, train.target
        int_cols = [0, 1, 2, 3, 4, 5, 6, 7]  # col 6 ... working day ... linearly dependent on others?
        real_cols = [8, 9, 10, 11]
        generator = np.random.RandomState(args.seed)

        # hyper-parameters:
        test_size = 0.1
        poly_max = 2
        epoch_num = 2000
        batch_size = 10
        learning_rate = 0.005
        norm_strength = 3
        norm_strength_sgd = 0.005

        X_train, X_test, y_train, y_test = \
            sklearn.model_selection.train_test_split(data, target, test_size=test_size, random_state=args.seed)

        enc = sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore")
        scaler = sklearn.preprocessing.StandardScaler()
        tr = sklearn.compose.ColumnTransformer([("enc1", enc, int_cols), ("scaler1", scaler, real_cols)])
        poly = sklearn.preprocessing.PolynomialFeatures(poly_max)  # include_bias=False

        pipe = sklearn.pipeline.Pipeline([('trans', tr), ('poly_feat', poly)])
        pipe.fit(X_train)

        X_train_reg = pipe.transform(X_train)
        X_test_reg = pipe.transform(X_test)
        #print(X_train_reg.shape)

        weights = generator.uniform(size=X_train_reg.shape[1], low=-0.1, high=0.1)
        print("rmse with rnd weights:",
              sklearn.metrics.mean_squared_error(y_train, X_train_reg @ weights, squared=False))

        rmses = []

        for epoch in range(epoch_num):
            permutation = generator.permutation(X_train_reg.shape[0])
            for i in range(X_train_reg.shape[0] // batch_size):
                grad_sum = np.zeros(X_train_reg.shape[1])
                for j in range(batch_size):
                    index = permutation[i * batch_size + j]
                    grad = (np.transpose(X_train_reg[index]) @ weights - y_train[index]) * X_train_reg[index]
                    grad_sum += grad
                grad_sum = grad_sum / batch_size
                weights = weights - learning_rate * (grad_sum + norm_strength_sgd * weights)
            rmses.append(sklearn.metrics.mean_squared_error(y_test, X_test_reg @ weights, squared=False))



        #print(rmses)
        #print("min rmse from sgd:", min(rmses))
        #pred_test = X_test_reg @ weights
        test_rmse = sklearn.metrics.mean_squared_error(y_test, X_test_reg @ weights, squared=False)
        print(test_rmse)

        #model = sklearn.linear_model.LinearRegression().fit(X_train_reg, y_train)

        lambdas = np.geomspace(0.01, 10, num=300)
        rmses = []

        """
        for lam in lambdas:
            #print(lam)
            lin_reg_lam = sklearn.linear_model.Ridge(alpha=lam).fit(X_train_reg, y_train)
            model_lam = sklearn.pipeline.Pipeline([("trans", pipe), ("lin_reg", lin_reg_lam)])
            predict = model_lam.predict(X_test)
            rmses.append(sklearn.metrics.mean_squared_error(y_test, predict, squared=False))
        print("min je:", min(rmses))
        """


        lin_reg = sklearn.linear_model.Ridge(alpha=norm_strength).fit(X_train_reg, y_train)
        lin_reg_sgd = sklearn.linear_model.Ridge().fit(X_train_reg, y_train)
        print(lin_reg_sgd.coef_)
        lin_reg_sgd.coef_ = weights
        print(lin_reg_sgd.coef_)
        model = sklearn.pipeline.Pipeline([("trans", pipe), ("lin_reg", lin_reg)])
        model_sgd = sklearn.pipeline.Pipeline([("trans", pipe), ("lin_reg", lin_reg_sgd)])

        pred2 = model_sgd.predict(X_test)
        pred = model.predict(X_test)
        explicit_rmse = sklearn.metrics.mean_squared_error(y_test, pred, squared=False)
        explicit_rmse2 = sklearn.metrics.mean_squared_error(y_test, pred2, squared=False)
        print("ridge rmse, sgd rmse:", explicit_rmse, explicit_rmse2)



        # TODO: Train a model on the given dataset and store it in `model`.
        #model = ...

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        #predictions = ...

        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)