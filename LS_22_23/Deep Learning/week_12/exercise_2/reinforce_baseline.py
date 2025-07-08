#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import gymnasium as gym
import numpy as np
import tensorflow as tf

import wrappers

# 3d76595a-e687-11e9-9ce9-00505601122b
# 594215cf-e687-11e9-9ce9-00505601122b

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.") #False
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=15, type=int, help="Random seed.") #None, 84 spatne
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")
parser.add_argument("--episodes", default=200, type=int, help="Training episodes.")
parser.add_argument("--hidden_layer_size1", default=128, type=int, help="Size of hidden layer.")
parser.add_argument("--hidden_layer_size2", default=128, type=int, help="Size of hidden layer.")
parser.add_argument("--hidden_layer_size3", default=42, type=int, help="Size of hidden layer.")
parser.add_argument("--hidden_layer_size_baseline", default=128, type=int, help="Size of hidden layer baseline.")
parser.add_argument("--learning_rate", default=0.0061, type=float, help="Learning rate.")


class Agent:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create a suitable model. The predict method assumes
        # the policy network is stored as `self._model`.
        #
        # Apart from the model defined in `reinforce`, define also another
        # model for computing the baseline (with a single output without an activation).
        # (Alternatively, this baseline computation can be grouped together
        # with the policy computation in a single `tf.keras.Model`.)
        #
        # Using Adam optimizer with given `args.learning_rate` for both models
        # is a good default.

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=env.observation_space.shape),
            tf.keras.layers.Dense(args.hidden_layer_size1, activation=tf.nn.relu),
            #tf.keras.layers.Dense(args.hidden_layer_size2, activation=tf.nn.relu),
            tf.keras.layers.Dense(args.hidden_layer_size3, activation=tf.nn.relu),
            tf.keras.layers.Dense(2, activation=tf.nn.softmax)])

        model.compile(
            optimizer=tf.optimizers.Adam(jit_compile=False, learning_rate=args.learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy()
        )

        self._model = model

        baseline = tf.keras.Sequential([
            tf.keras.layers.Input(shape=env.observation_space.shape),
            tf.keras.layers.Dense(args.hidden_layer_size_baseline, activation=tf.nn.relu),
            #tf.keras.layers.Dense(args.hidden_layer_size2, activation=tf.nn.relu),
            tf.keras.layers.Dense(1)])

        baseline.compile(
            optimizer=tf.optimizers.Adam(jit_compile=False, learning_rate=args.learning_rate),
            loss=tf.keras.losses.MeanSquaredError()
        )

        self._model = model
        self._baseline = baseline


        #raise NotImplementedError()

    # Define a training method.
    #
    # Note that we need to use `raw_tf_function` (a faster variant of `tf.function`)
    # and manual `tf.GradientTape` for efficiency (using `fit` or `train_on_batch`
    # on extremely small batches has considerable overhead).
    @wrappers.raw_tf_function(dynamic_dims=1)
    def train(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:
        # TODO: Perform training, using the loss from the REINFORCE with baseline
        # algorithm. You should:
        # - compute the predicted baseline using the baseline model
        # - train the baseline model to predict `returns`
        # - train the policy model, using `returns - predicted_baseline` as
        #   the advantage estimate

        with tf.GradientTape() as tape1:
            pred_baseline = self._baseline(states, training=True)
            delta = tf.expand_dims(returns, axis=-1) - tf.cast(pred_baseline, dtype=tf.float64)
            #delta = returns - tf.cast(pred_baseline[:, 0], dtype=tf.float64)
            loss_baseline = self._baseline.compute_loss(states, returns, pred_baseline)


        with tf.GradientTape() as tape2:
            prob = self._model(states, training=True)
            loss = self._model.compute_loss(states, tf.one_hot(actions, 2), prob, sample_weight=delta)

        self._model.optimizer.minimize(loss, self._model.trainable_variables, tape=tape2)
        self._baseline.optimizer.minimize(loss_baseline, self._baseline.trainable_variables, tape=tape1)


        #raise NotImplementedError()

    # Predict method, again with the `raw_tf_function` for efficiency.
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict(self, states: np.ndarray) -> np.ndarray:
        return self._model(states)


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        # tf.data.experimental.enable_debug_mode()

    # Construct the agent
    agent = Agent(env, args)

    # Training
    for _ in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform episode
            states, actions, rewards = [], [], []
            state, done = env.reset()[0], False
            while not done:
                # TODO(reinforce): Choose `action` according to probabilities
                # distribution (see `np.random.choice`), which you
                # can compute using `agent.predict` and current `state`.
                #action = ...

                prob = agent.predict(np.expand_dims(state, axis=0))[0]
                action = np.random.choice(a=env.action_space.n, p=prob)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # TODO(reinforce): Compute returns from the received rewards

            returns = []
            for i in range(0, len(rewards)):
                one_return = np.sum(rewards[i:])
                returns.append(one_return)

            #returns = tf.cumsum(rewards, reverse=True).numpy().tolist()

            # TODO(reinforce): Add states, actions and returns to the training batch

            batch_states += states
            batch_actions += actions
            batch_returns += returns

        # TODO(reinforce): Train using the generated batch.

        agent.train(np.array(batch_states), np.array(batch_actions), np.array(batch_returns))
        #agent.train(tf.convert_to_tensor(batch_states), tf.convert_to_tensor(batch_actions), tf.convert_to_tensor(batch_returns))

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO(reinforce): Choose a greedy action
            #action = ...

            prob = agent.predict(np.expand_dims(state, axis=0))[0]
            action = np.argmax(prob)

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("CartPole-v1"), args.seed, args.render_each)

    main(env, args)
