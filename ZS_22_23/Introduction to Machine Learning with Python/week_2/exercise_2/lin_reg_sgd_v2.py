#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--epochs", default=50, type=int, help="Number of SGD training epochs")
parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization strength")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace):  # -> tuple[list[float], float, float]:
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate an artificial regression dataset
    data, target = sklearn.datasets.make_regression(n_samples=args.data_size, random_state=args.seed)
    #print(data, data.shape, target, target.shape)

    # TODO: Append a constant feature with value 1 to the end of every input data
    sample_size, feature_size = data.shape
    #print("sample_size, feature_size", sample_size, feature_size)
    new_data = np.concatenate((data, np.ones((sample_size, 1))), axis=1)
    #print(new_data)

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target \
        = sklearn.model_selection.train_test_split(new_data, target, test_size=args.test_size, random_state=args.seed)
    print(train_data.shape, train_target.shape)

    # Generate initial linear regression weights
    weights = generator.uniform(size=train_data.shape[1], low=-0.1, high=0.1)
    print(weights.shape)

    train_rmses, test_rmses = [], []
    #for epoch in range(0):
    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])
        #print(permutation)
        #print("number of batches:", train_data.shape[0]//args.batch_size)

        for i in range(train_data.shape[0]//args.batch_size):
            #print("batch size:", args.batch_size)
            #print("data and target shape:", train_data.shape, train_target.shape)
            #print("batch loop:", i)
            #index = permutation[i]
            grad_sum = np.zeros(train_data.shape[1])
            for j in range(args.batch_size):
                index = permutation[i*args.batch_size+j]
                #print(index)
                #print("index, batch_size, j, multip", index, args.batch_size, j, index * args.batch_size + j)
                grad = (np.transpose(train_data[index]) @ weights - train_target[index]) * train_data[index]
                #print("grad", grad)
                grad_sum += grad
            #print(grad_sum[0])
            grad_sum = grad_sum/(args.batch_size)
            #print(grad_sum[0])
            weights = weights - args.learning_rate * (grad_sum + args.l2 * weights)
            #print(weights)
        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # A gradient for input example $(x_i, t_i)$ is $(x_i^T weights - t_i) x_i$,
        # and the SGD update is
        #   weights = weights - args.learning_rate * (gradient + args.l2 * weights)`.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.

        pred_train = train_data @ weights
        pred_test = test_data @ weights
        train_rmses.append(sklearn.metrics.mean_squared_error(train_target, pred_train, squared=False))
        test_rmses.append(sklearn.metrics.mean_squared_error(test_target, pred_test, squared=False))

        # TODO: Append current RMSE on train/test to `train_rmses`/`test_rmses`.

    # TODO: Compute into `explicit_rmse` test data RMSE when fitting
    # `sklearn.linear_model.LinearRegression` on `train_data` (ignoring `args.l2`).
    model = sklearn.linear_model.LinearRegression().fit(train_data, train_target)
    pred = model.predict(test_data)
    explicit_rmse = sklearn.metrics.mean_squared_error(test_target, pred, squared=False)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(train_rmses, label="Train")
        plt.plot(test_rmses, label="Test")
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        plt.legend()
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return weights, test_rmses[-1], explicit_rmse


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, sgd_rmse, explicit_rmse = main(args)
    print("Test RMSE: SGD {:.2f}, explicit {:.2f}".format(sgd_rmse, explicit_rmse))
    print("Learned weights:", *("{:.2f}".format(weight) for weight in weights[:12]), "...")