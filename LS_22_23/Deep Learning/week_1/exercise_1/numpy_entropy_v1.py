#!/usr/bin/env python3

# Super-duper tym
# Pepa z depa: 3d76595a-e687-11e9-9ce9-00505601122b
# Uzasna Berenika co vsechno vymyslela: 594215cf-e687-11e9-9ce9-00505601122b
import argparse
from typing import Tuple

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_path", default="numpy_entropy_data_2.txt", type=str, help="Data distribution path.")
parser.add_argument("--model_path", default="numpy_entropy_model_2.txt", type=str, help="Model distribution path.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> Tuple[float, float, float]:
    # TODO: Load data distribution, each line containing a datapoint -- a string.
    letters = {}
    with open(args.data_path, "r") as data:
        counter = 0
        for line in data:
            counter += 1
            line = line.rstrip("\n")
            # TODO: Process the line, aggregating data with built-in Python
            # data structures (not NumPy, which is not suitable for incremental
            # addition and string mapping).

            if line in letters:
                letters[line] += 1
            else:
                letters[line] = 1


    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. Alternatively,
    # the NumPy array might be created after loading the model distribution.

    probs = {}
    for x in letters.keys():
        probs[x] = 0

    # TODO: Load model distribution, each line `string \t probability`.
    with open(args.model_path, "r") as model:
        for line in model:
            line = line.rstrip("\n")

            # TODO: Process the line, aggregating using Python data structures.
            line = line.split()
            probs[line[0]] = float(line[1])
            if line[0] not in letters.keys():
                letters[line[0]] = 0

    # TODO: Create a NumPy array containing the model distribution.

    d_distr = np.array([letters[i] for i in sorted(list(letters.keys()))]) / counter
    m_distr = np.array([probs[i] for i in sorted(list(probs.keys()))])

    # TODO: Compute the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).

    entropy = - np.sum(np.log(d_distr[d_distr > 0]) * d_distr[d_distr > 0])

    # TODO: Compute cross-entropy H(data distribution, model distribution).
    # When some data distribution elements are missing in the model distribution,
    # return `np.inf`.

    crossentropy = - np.sum(np.log(m_distr) * d_distr)
    if crossentropy == np.nan:
        crossentropy = np.inf


    # TODO: Compute KL-divergence D_KL(data distribution, model_distribution),
    # again using `np.inf` when needed.
    kl_divergence = crossentropy - entropy

    # Return the computed values for ReCodEx to validate.
    return entropy, crossentropy, kl_divergence


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(args)
    print("Entropy: {:.2f} nats".format(entropy))
    print("Crossentropy: {:.2f} nats".format(crossentropy))
    print("KL divergence: {:.2f} nats".format(kl_divergence))