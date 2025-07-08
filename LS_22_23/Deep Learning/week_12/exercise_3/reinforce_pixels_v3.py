#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import gymnasium as gym
import numpy as np
import tensorflow as tf

import cart_pole_pixels_environment
import wrappers

# 3d76595a-e687-11e9-9ce9-00505601122b optimista
# 594215cf-e687-11e9-9ce9-00505601122b pesimista

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--debug", default=True, action="store_true", help="If given, run functions eagerly.") #False
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")  # None
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=5, type=int, help="Batch size.")
parser.add_argument("--episodes", default=50, type=int, help="Training episodes.")  # 5000
parser.add_argument("--hidden_layer_size1", default=70, type=int, help="Size of hidden layer.")
#parser.add_argument("--hidden_layer_size2", default=70, type=int, help="Size of hidden layer 2.")
#parser.add_argument("--hidden_layer_size3", default=200, type=int, help="Size of hidden layer 3.")
parser.add_argument("--hidden_layer_size_baseline", default=70, type=int, help="Size of hidden layer baseline.")
parser.add_argument("--learning_rate", default=0.0005, type=float, help="Learning rate.")

class Agent:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO*: Create a suitable model. The predict method assumes
        # it is stored as `self._model`.
        #
        # Using Adam optimizer with given `args.learning_rate` is a good default.

        """"
        inputs = tf.keras.layers.Input(shape=env.observation_space.shape)
        #hidden = tf.keras.layers.Conv2D(4, 3, 2, padding="valid", activation=tf.nn.relu)(inputs)
        hidden = tf.keras.layers.Conv2D(4, 3, 2, padding="same")(inputs)
        hidden = tf.keras.layers.LayerNormalization()(hidden)
        hidden = tf.nn.relu(hidden)

        #hidden = tf.keras.layers.Conv2D(8, 3, 2, padding="valid", activation=tf.nn.relu)(hidden)
        hidden = tf.keras.layers.Conv2D(8, 3, 2, padding="same")(hidden)
        hidden = tf.keras.layers.LayerNormalization()(hidden)
        hidden = tf.nn.relu(hidden)
        #hidden = tf.keras.layers.Conv2D(16, 3, 2, padding="valid", activation=tf.nn.relu)(hidden)
        hidden = tf.keras.layers.Conv2D(16, 3, 2, padding="same")(hidden)
        hidden = tf.nn.relu(hidden)
        hidden = tf.keras.layers.LayerNormalization()(hidden)
        hidden = tf.keras.layers.Conv2D(20, 3, 2, padding="same")(hidden)
        hidden = tf.nn.relu(hidden)
        hidden = tf.keras.layers.LayerNormalization()(hidden)
        hidden = tf.keras.layers.Flatten()(hidden)
        hidden = tf.keras.layers.Dense(args.hidden_layer_size1, activation=tf.nn.relu)(hidden)
        outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(hidden)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        """

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=env.observation_space.shape),
            tf.keras.layers.Conv2D(5, 3, 2, padding="same", activation=tf.nn.relu, use_bias=False),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv2D(10, 3, 2, padding="same", activation=tf.nn.relu, use_bias=False),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv2D(15, 3, 2, padding="same", activation=tf.nn.relu, use_bias=False),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv2D(20, 3, 2, padding="same", activation=tf.nn.relu, use_bias=False),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv2D(25, 3, 2, padding="same", activation=tf.nn.relu, use_bias=False),
            tf.keras.layers.LayerNormalization(),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(args.hidden_layer_size1, activation=tf.nn.relu),
            tf.keras.layers.Dense(2, activation=tf.nn.softmax)])

        model.compile(
            optimizer=tf.optimizers.Adam(jit_compile=False, learning_rate=args.learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy()
            )

        baseline = tf.keras.Sequential([
            tf.keras.layers.Input(shape=env.observation_space.shape),
            tf.keras.layers.Conv2D(4, 3, 2, padding="same", activation=tf.nn.relu, use_bias=False),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv2D(10, 3, 2, padding="same",  activation=tf.nn.relu, use_bias=False),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv2D(15, 3, 2, padding="same", activation=tf.nn.relu, use_bias=False),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv2D(20, 3, 2, padding="same", activation=tf.nn.relu, use_bias=False),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv2D(25, 3, 2, padding="same", activation=tf.nn.relu, use_bias=False),
            tf.keras.layers.LayerNormalization(),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(args.hidden_layer_size_baseline, activation=tf.nn.relu),
            # tf.keras.layers.Dense(args.hidden_layer_size2, activation=tf.nn.relu),
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
        # TODO*: Perform training, using the loss from the REINFORCE algorithm.
        # The easiest approach is to use the `sample_weight` argument of
        # the `__call__` method of a suitable subclass of `tf.losses.Loss`,
        # but you can also construct a loss instance with `reduction=tf.losses.Reduction.NONE`
        # and perform the weighting manually.

        with tf.GradientTape() as tape1:
            pred_baseline = self._baseline(states, training=True)
            delta = tf.expand_dims(returns, axis=-1) - tf.cast(pred_baseline, dtype=tf.float64)
            #loss_baseline = self._model.compute_loss(states, returns, pred_baseline, sample_weight=delta)
            loss_baseline = self._model.compute_loss(states, returns, pred_baseline)

        #self._baseline.optimizer.minimize(loss_baseline, self._baseline.trainable_variables, tape=tape1)

        with tf.GradientTape() as tape2:
            prob = self._model(states, training=True)
            loss = self._model.compute_loss(states, tf.one_hot(actions, 2), prob, sample_weight=delta)

        self._model.optimizer.minimize(loss, self._model.trainable_variables, tape=tape2)
        self._baseline.optimizer.minimize(loss_baseline, self._baseline.trainable_variables, tape=tape1)

        #print(self._model.trainable_variables)

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

    if args.recodex:
        # TODO: Load a pre-trained agent and evaluate it.
        model = tf.keras.models.load_model("reinforce_pixels_v3.h5")
        while True:
            state, done = env.reset(start_evaluation=True)[0], False
            while not done:
                # TODO: Choose an action
                #action = ...
                #action = np.argmax(model.predict(np.expand_dims(state, axis=0))[0])
                action = tf.argmax(model(tf.convert_to_tensor(state)[tf.newaxis, :])[0]).numpy()
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
    else:

        # TODO: Perform training
        #raise NotImplementedError()

        agent = Agent(env, args)

        for _ in range(args.episodes // args.batch_size):
            batch_states, batch_actions, batch_returns = [], [], []
            for _ in range(args.batch_size):
                # Perform episode
                states, actions, rewards = [], [], []
                state, done = env.reset()[0], False
                while not done:
                    # TODO*: Choose `action` according to probabilities
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

                # TODO*: Compute returns from the received rewards

                returns = []
                for i in range(0, len(rewards)):
                    one_return = np.sum(rewards[i:])
                    returns.append(one_return)


                # TODO*: Add states, actions and returns to the training batch

                batch_states += states
                batch_actions += actions
                batch_returns += returns

            # TODO*: Train using the generated batch.

            agent.train(np.array(batch_states), np.array(batch_actions), np.array(batch_returns))

        tf.keras.Model.save(agent._model, filepath="reinforce_pixels_v3.h5", include_optimizer=False)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("CartPolePixels-v1"), args.seed, args.render_each)

    main(env, args)