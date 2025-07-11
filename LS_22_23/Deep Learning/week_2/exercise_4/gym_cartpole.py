#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import Optional
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

# 3d76595a-e687-11e9-9ce9-00505601122b optimista
# 594215cf-e687-11e9-9ce9-00505601122b pesimista

import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--evaluate", default=True, action="store_true", help="Evaluate the given model")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--render", default=False, action="store_true", help="Render during evaluation")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")
parser.add_argument("--epochs", default=150, type=int, help="Number of epochs.")
parser.add_argument("--model", default="gym_cartpole_model.h5", type=str, help="Output model path.")

parser.add_argument("--hidden_layer", default=300, type=int, help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.0005, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=0.00005, type=float, help="Final learning rate.") #None
parser.add_argument("--momentum", default=None, type=float, help="Nesterov momentum to use in SGD.")
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adam"], help="Optimizer to use.") #SGD
parser.add_argument("--decay", default="exponential", choices=["linear", "exponential", "cosine"], help="Decay type") #None


def evaluate_model(
    model: tf.keras.Model, seed: int = 42, episodes: int = 100, render: bool = False, report_per_episode: bool = False
) -> float:
    """Evaluate the given model on CartPole-v1 environment.

    Returns the average score achieved on the given number of episodes.
    """
    import gymnasium as gym

    # Create the environment
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    env.reset(seed=seed)

    # Evaluate the episodes
    total_score = 0
    for episode in range(episodes):
        observation, score, done = env.reset()[0], 0, False
        while not done:
            prediction = model(observation[np.newaxis])[0].numpy()
            if len(prediction) == 1:
                action = 1 if prediction[0] > 0.5 else 0
            elif len(prediction) == 2:
                action = np.argmax(prediction)
            else:
                raise ValueError("Unknown model output shape, only 1 or 2 outputs are supported")

            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            done = terminated or truncated

        total_score += score
        if report_per_episode:
            print("The episode {} finished with score {}.".format(episode + 1, score))
    return total_score / episodes


def main(args: argparse.Namespace) -> Optional[tf.keras.Model]:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    if not args.evaluate:
        # Create logdir name
        args.logdir = os.path.join("logs", "{}-{}-{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
        ))

        # Load the data
        data = np.loadtxt("gym_cartpole_data.txt")
        observations, labels = data[:, :-1], data[:, -1].astype(np.int32)

        # TODO: Create the model in the `model` variable. Note that
        # the model can perform any of:
        # - binary classification with 1 output and sigmoid activation;
        # - two-class classification with 2 outputs and softmax activation.
        model = tf.keras.Sequential([
        #tf.keras.layers.Flatten(input_shape=[MNIST.H, MNIST.W, MNIST.C]),
        tf.keras.layers.Input([4]),
        tf.keras.layers.Dense(args.hidden_layer, activation=tf.nn.relu),
        tf.keras.layers.Dense(2, activation=tf.nn.sigmoid),
        ])

        if args.momentum is not None:
            nest = True
            mom = args.momentum
        else:
            nest = False
            mom = 0.0

        steps = args.epochs * observations.shape[0] // args.batch_size

        if args.decay == "linear":
            lr = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=args.learning_rate,
                decay_steps=steps,
                end_learning_rate=args.learning_rate_final,
                power=1.0)
        elif args.decay == "exponential":
            lr = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=args.learning_rate,
                decay_steps=steps,
                decay_rate=args.learning_rate_final / args.learning_rate,
                staircase=False)
        elif args.decay == "cosine":
            lr = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=args.learning_rate,
                decay_steps=steps,
                alpha=args.learning_rate_final / args.learning_rate)
        else:
            lr = args.learning_rate

        if args.optimizer == "SGD":
            opt = tf.optimizers.SGD(nesterov=nest, momentum=mom, learning_rate=lr)
        else:
            opt = tf.optimizers.Adam(learning_rate=lr)


        # TODO: Prepare the model for training using the `model.compile` method.
        model.compile(
            optimizer=opt,
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy("accuracy")],
        )

        #tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1)

        model.fit(
            observations, labels, batch_size=args.batch_size, epochs=args.epochs, #callbacks=[tb_callback]
        )

        # Save the model, without the optimizer state.
        model.save(args.model, include_optimizer=False)

    else:
        # Evaluating, either manually or in ReCodEx
        model = tf.keras.models.load_model(args.model, compile=False)

        if args.recodex:
            return model
        else:
            score = evaluate_model(model, seed=args.seed, render=args.render, report_per_episode=True)
            print("The average score was {}.".format(score))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
