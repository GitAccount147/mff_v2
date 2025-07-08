#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--C", default=1, type=float, help="Inverse regularization strength")
parser.add_argument("--classes", default=5, type=int, help="Number of classes")  # 10
parser.add_argument("--kernel", default="poly", type=str, help="Kernel type (poly/rbf)")
parser.add_argument("--kernel_degree", default=2, type=int, help="Degree for poly kernel")  # 1
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--max_iterations", default=20, type=int, help="Maximum number of iterations to perform")  # 1000
parser.add_argument("--max_passes_without_as_changing", default=10, type=int, help="Stopping condition")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.8, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")  # 0.5
parser.add_argument("--tolerance", default=1e-7, type=float, help="Default tolerance for KKT conditions")
# If you add more arguments, ReCodEx will keep them with your default values.


def kernel(args: argparse.Namespace, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    # TODO: Use the kernel from the smo_algorithm assignment.
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


def smo(
    args: argparse.Namespace,
    train_data: np.ndarray, train_target: np.ndarray,
    test_data: np.ndarray, test_target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, list[float], list[float]]:
    # TODO: Use the SMO algorithm from the smo_algorithm assignment.
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


def main(args: argparse.Namespace) -> float:
    # Load the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)
    data = sklearn.preprocessing.MinMaxScaler().fit_transform(data)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Using One-vs-One scheme, train (K \binom 2) classifiers, one for every
    # pair of classes $i < j$, using the `smo` method.
    #
    # When training a classifier for classes $i < j$:
    # - keep only the training data of these classes, in the same order
    #   as in the input dataset;
    # - use targets 1 for the class $i$ and -1 for the class $j$.

    classifiers = []

    for i in range(args.classes):
        for j in range(i + 1, args.classes):
            print("Training classes {} and {}".format(i, j))
            new_train_data, new_train_target = [], []
            for k in range(len(train_data)):
                if train_target[k] == i:
                    new_train_data.append(train_data[k])
                    new_train_target.append(1)
                if train_target[k] == j:
                    new_train_data.append(train_data[k])
                    new_train_target.append(-1)
            new_train_data = np.array(new_train_data)
            new_train_target = np.array(new_train_target)

            new_test_data, new_test_target = [], []
            for k in range(len(test_data)):
                if test_target[k] == i:
                    new_test_data.append(test_data[k])
                    new_test_target.append(1)
                if test_target[k] == j:
                    new_test_data.append(test_data[k])
                    new_test_target.append(-1)
            new_test_data = np.array(new_test_data)
            new_test_target = np.array(new_test_target)

            support_vectors, support_vector_weights, bias, train_acc, test_acc = smo(
                args, new_train_data, new_train_target, new_test_data, new_test_target)

            classifiers.append((i, j, support_vectors, support_vector_weights, bias))

    votes = np.zeros((test_data.shape[0], args.classes))
    for i in range(len(classifiers)):
        classifier = classifiers[i]
        class_i, class_j, support_vectors, support_vector_weights, bias = classifier

        ker = kernel(args, support_vectors, test_data)
        result = np.sign(support_vector_weights.T @ ker + bias)
        for j in range(len(result)):
            if result[j] == 1:
                votes[j][class_i] += 1
            else:
                votes[j][class_j] += 1

    final_result = np.zeros((test_data.shape[0], 1))
    for i in range(test_data.shape[0]):
        final_result[i][0] = np.argmax(votes[i])

    test_accuracy = sklearn.metrics.accuracy_score(test_target, final_result)

    # TODO: Classify the test set by majority voting of all the trained classifiers,
    # using the lowest class index in the case of ties.
    #
    # Note that during prediction, only the support vectors returned by the `smo`
    # should be used, not all training data.
    #
    # Finally, compute the test set prediction accuracy.

    return test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(args)
    print("Test set accuracy: {:.2f}%".format(100 * accuracy))