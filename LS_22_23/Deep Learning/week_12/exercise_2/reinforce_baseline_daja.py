#!/usr/bin/env python3
#
#team:
#Pepa
# 59014ac0-e687-11e9-9ce9-00505601122b
#Lucka
# b4fb3099-e69b-11e9-9ce9-00505601122b
#Dája
# 45e69378-e687-11e9-9ce9-00505601122b
#
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import gymnasium as gym
import numpy as np
import tensorflow as tf

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=15, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")
parser.add_argument("--episodes", default=200, type=int, help="Training episodes.")
parser.add_argument("--hidden_layer_size", default=128, type=int, help="Size of hidden layer.")
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
        ## model
        inputs = tf.keras.layers.Input(shape=[4])
        l = inputs
        l = tf.keras.layers.Dense(args.hidden_layer_size, activation=tf.nn.relu)(l)
        l = tf.keras.layers.Dense(args.hidden_layer_size//3, activation=tf.nn.relu)(l)
        l = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(l)
        model = tf.keras.Model(inputs=inputs, outputs=l)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=tf.losses.CategoricalCrossentropy(),
            metrics=[tf.metrics.CategoricalAccuracy("accuracy")],
        )
        self._model = model
        ## baseline
        inputs = tf.keras.layers.Input(shape=[4])
        l = inputs
        l = tf.keras.layers.Dense(args.hidden_layer_size, activation=tf.nn.relu)(l)
        l = tf.keras.layers.Dense(1, activation=None)(l)
        baseline = tf.keras.Model(inputs=inputs, outputs=l)
        baseline.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=tf.keras.losses.MeanSquaredError(),
        )
        self._baseline_regressor = baseline
        #
        self._lr = args.learning_rate

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
        #CCE = tf.keras.losses.CategoricalCrossentropy()
        CCE = self._model.loss
        MSE = self._baseline_regressor.loss
        lr = self._lr
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        returns = tf.convert_to_tensor(returns)
        with tf.GradientTape() as tape0:
            baseline = self._baseline_regressor(states)
            aerr  = MSE(y_true=returns, y_pred=baseline)
        with tf.GradientTape() as tape:
            prob = self._model(states)
            loss = CCE(y_true=tf.one_hot(actions, depth=2), y_pred=prob, sample_weight=returns-baseline[:,0])
        variables = self._model.variables
        self._model.optimizer.minimize(loss, variables, tape)
        bvariables = self._baseline_regressor.variables
        self._baseline_regressor.optimizer.minimize(aerr, bvariables, tape0)

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
    rets = []
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
                action = np.random.choice([0,1],p=agent.predict([state])[0])

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state
            else:
                rets.append(sum(rewards))

            # TODO(reinforce): Compute returns from the received rewards
            returns = tf.cumsum(rewards, reverse=True).numpy().tolist()

            # TODO(reinforce): Add states, actions and returns to the training batch
            batch_states += states
            batch_actions += actions
            batch_returns += returns

        # TODO(reinforce): Train using the generated batch.
        if len(rets) > 10 and np.mean(rets[-10:]) > 490:
            break
        agent.train(states, actions, returns)

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO(reinforce): Choose a greedy action
            action = tf.argmax(agent.predict([state])[0]).numpy()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("CartPole-v1"), args.seed, args.render_each)

    main(env, args)
