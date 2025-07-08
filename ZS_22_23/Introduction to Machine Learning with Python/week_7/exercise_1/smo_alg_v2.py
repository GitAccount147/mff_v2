#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--C", default=1, type=float, help="Inverse regularization strength")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--kernel", default="rbf", type=str, help="Kernel type (poly/rbf)")
parser.add_argument("--kernel_degree", default=1, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--max_iterations", default=1000, type=int, help="Maximum number of iterations to perform")
parser.add_argument("--max_passes_without_as_changing", default=10, type=int, help="Stopping condition")
parser.add_argument("--plot", default=True, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
parser.add_argument("--tolerance", default=1e-7, type=float, help="Default tolerance for KKT conditions")
# If you add more arguments, ReCodEx will keep them with your default values.


def kernel(args: argparse.Namespace, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    # TODO: As in `kernel_linear_regression`, We consider the following `args.kernel`s:
    # - "poly": K(x, z; degree, gamma) = (gamma * x^T z + 1) ^ degree
    # - "rbf": K(x, z; gamma) = exp^{- gamma * ||x - z||^2}

    if len(z.shape) == 1:
        z = z.reshape(1, -1)

    result = np.zeros((x.shape[0], z.shape[0]))

    if args.kernel == "poly":
        gram = x @ np.transpose(z)
        for i in range(x.shape[0]):
            for j in range(z.shape[0]):
                result[i][j] = np.power(args.kernel_gamma * gram[i][j] + 1, args.kernel_degree)
    else:
        for i in range(x.shape[0]):
            for j in range(z.shape[0]):
                result[i][j] = np.exp(- args.kernel_gamma * np.power(np.linalg.norm(x[i] - z[j]), 2))
    return result


# We implement the SMO algorithm as a separate method, so we can use
# it in the svm_multiclass assignment too.
def smo(
    args: argparse.Namespace,
    train_data: np.ndarray, train_target: np.ndarray,
    test_data: np.ndarray, test_target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, list[float], list[float]]:
    # Create initial weights.
    a, b = np.zeros(len(train_data)), 0
    generator = np.random.RandomState(args.seed)

    # generate kernels:
    ker = kernel(args, train_data, train_data)
    ker_test = kernel(args, train_data, test_data)

    passes_without_as_changing = 0
    train_accs, test_accs = [], []
    for _ in range(args.max_iterations):
        as_changed = 0
        # Iterate through the data.
        for i, j in enumerate(generator.randint(len(a) - 1, size=len(a))):
            # We want j != i, so we "skip" over the value of i.
            j = j + (j >= i)

            # TODO: Check that a[i] fulfills the KKT conditions, using `args.tolerance` during comparisons.

            # create better notation:
            a_i, a_j = a[i], a[j]
            t_i, t_j = train_target[i], train_target[j]
            C, tol = args.C, args.tolerance
            ker_i, ker_j, ker_ij, ker_ji = ker[i][i], ker[j][j], ker[i][j], ker[j][i]

            kkt_fulfilled = True
            pred_i = np.dot(a * train_target, ker[i]) + b
            pred_j = np.dot(a * train_target, ker[j]) + b
            E_i = pred_i - t_i
            E_j = pred_j - t_j

            if (a_i < C - tol and t_i * E_i < - tol) or (a_i > tol and t_i * E_i > tol):
                kkt_fulfilled = False

            derivative_small = False
            second_derivative = 2 * ker_ij - ker_i - ker_j

            if second_derivative > - tol:
                derivative_small = True

            if not kkt_fulfilled and not derivative_small:
                a_j_new = a_j - t_j * (E_i - E_j) * (1 / second_derivative)

                if t_i == t_j:
                    L, H = max(0, a_i + a_j - C), min(C, a_i + a_j)
                else:
                    L, H = max(0, a_j - a_i), min(C, C + a_j - a_i)

                clip_okay = True
                if a_j_new < L - tol:
                    a_j_new = L
                if a_j_new > H + tol:
                    a_j_new = H

                if abs(a_j_new - a_j) < tol:
                    clip_okay = False

                if clip_okay:
                    a_i_new = a_i - t_i * t_j * (a_j_new - a_j)
                    b_j_new = b - E_j - t_i * (a_i_new - a_i) * ker_ij - t_j * (a_j_new - a_j) * ker_j
                    b_i_new = b - E_i - t_i * (a_i_new - a_i) * ker_i - t_j * (a_j_new - a_j) * ker_ji

                    if tol < a_i_new < C - tol:
                        b_new = b_i_new
                    elif tol < a_j_new < C - tol:
                        b_new = b_j_new
                    else:
                        b_new = (b_i_new + b_j_new) / 2

                    b, a[j], a[i] = b_new, a_j_new, a_i_new
                    as_changed += 1

            # If the conditions do not hold, then:
            # - compute the updated unclipped a_j^new.
            #
            #   If the second derivative of the loss with respect to a[j]
            #   is > -`args.tolerance`, do not update a[j] and continue
            #   with next i.

            # - clip the a_j^new to suitable [L, H].
            #
            #   If the clipped updated a_j^new differs from the original a[j]
            #   by less than `args.tolerance`, do not update a[j] and continue
            #   with next i.

            # - update a[j] to a_j^new, and compute the updated a[i] and b.
            #
            #   During the update of b, compare the a[i] and a[j] to zero by
            #   `> args.tolerance` and to C using `< args.C - args.tolerance`.

            # - increase `as_changed`.

        # TODO: After each iteration, measure the accuracy for both the
        # train set and the test set and append it to `train_accs` and `test_accs`.

        train_prediction = ker.T @ (a * train_target) + b
        test_prediction = ker_test.T @ (a * train_target) + b
        train_prediction_precise = np.sign(train_prediction)
        test_prediction_precise = np.sign(test_prediction)

        train_acc = sklearn.metrics.accuracy_score(train_target, train_prediction_precise)
        test_acc = sklearn.metrics.accuracy_score(test_target, test_prediction_precise)

        train_accs.append(train_acc)
        test_accs.append(test_acc)

        # Stop training if `args.max_passes_without_as_changing` passes were reached.
        passes_without_as_changing = 0 if as_changed else passes_without_as_changing + 1
        if passes_without_as_changing >= args.max_passes_without_as_changing:
            break

        if len(train_accs) % 100 == 0 and len(train_accs) < args.max_iterations:
            print("Iteration {}, train acc {:.1f}%, test acc {:.1f}%".format(
                len(train_accs), 100 * train_accs[-1], 100 * test_accs[-1]))

    # TODO: Create an array of support vectors (in the same order in which they appeared
    # in the training data; to avoid rounding errors, consider a training example
    # a support vector only if a_i > `args.tolerance`) and their weights (a_i * t_i).
    support_vectors, support_vector_weights = [], []
    for i in range(train_data.shape[0]):
        if a[i] > args.tolerance:
            support_vectors.append(list(train_data[i]))
            support_vector_weights.append(a[i] * train_target[i])

    support_vectors = np.array(support_vectors)
    support_vector_weights = np.array(support_vector_weights)

    print("Done, iteration {}, support vectors {}, train acc {:.1f}%, test acc {:.1f}%".format(
        len(train_accs), len(support_vectors), 100 * train_accs[-1], 100 * test_accs[-1]))

    return support_vectors, support_vector_weights, b, train_accs, test_accs


def main(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float, list[float], list[float]]:
    # Generate an artificial regression dataset, with +-1 as targets.
    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)
    target = 2 * target - 1

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Run the SMO algorithm.
    support_vectors, support_vector_weights, bias, train_accs, test_accs = smo(
        args, train_data, train_target, test_data, test_target)

    if args.plot:
        import matplotlib.pyplot as plt

        def plot(predict, support_vectors):
            xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
            ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
            predictions = [[predict(np.array([x, y])) for x in xs] for y in ys]
            test_mismatch = np.sign([predict(x) for x in test_data]) != test_target
            plt.figure()
            plt.contourf(xs, ys, predictions, levels=0, cmap="RdBu")
            plt.contour(xs, ys, predictions, levels=[-1, 0, 1], colors="k", zorder=1)
            plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, marker="o", label="Train", cmap="RdBu", zorder=2)
            plt.scatter(support_vectors[:, 0], support_vectors[:, 1], marker="o", s=90, label="Support Vectors", c="#0d0")
            plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, marker="*", label="Test", cmap="RdBu", zorder=2)
            plt.scatter(test_data[test_mismatch, 0], test_data[test_mismatch, 1], marker="*", s=130, label="Test Errors", c="#ff0")
            plt.legend(loc="upper center", ncol=4)

        # If you want plotting to work (not required for ReCodEx), you need to
        # define `predict_function` computing SVM value `y(x)` for the given x.
        def predict_function(x):
            ker_test_plot = kernel(args, support_vectors, x)
            result = ker_test_plot.T @ support_vector_weights + bias
            return result[0]

        plot(predict_function, support_vectors)
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return support_vectors, support_vector_weights, bias, train_accs, test_accs


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)