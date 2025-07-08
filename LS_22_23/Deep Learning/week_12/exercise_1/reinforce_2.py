#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import gymnasium as gym
import numpy as np
import tensorflow as tf

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--debug", default=True, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=3, type=int, help="Batch size.")
parser.add_argument("--episodes", default=7, type=int, help="Training episodes.")
parser.add_argument("--hidden_layer_size", default=100, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")


class Agent:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create a suitable model. The predict method assumes
        # it is stored as `self._model`.
        #
        # Using Adam optimizer with given `args.learning_rate` is a good default.
        #raise NotImplementedError()
        model = tf.keras.Sequential([
            tf.keras.layers.Input([4]),
            #tf.keras.layers.Input([None, args.batch_size]),
            #tf.keras.layers.Input([args.batch_size, 4]),
            #tf.keras.layers.Input([args.batch_size]),
            tf.keras.layers.Dense(args.hidden_layer_size, activation=tf.nn.relu),
            #tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
            #tf.keras.layers.Dense(2, activation=tf.nn.sigmoid),
            tf.keras.layers.Dense(2, activation=tf.nn.softmax),
        ])


        #self._model = tf.keras.Sequential()
        self._model = model

        inputs = tf.keras.layers.Input(shape=[None, 4])

        #activation_1 = tf.nn.softmax
        #dense = tf.keras.layers.Dense(units=args.hidden_layer_size, activation=activation_1)

        activation_2 = tf.nn.sigmoid
        outputs = tf.keras.layers.Dense(units=2, activation=activation_2)

        #self._model.add(inputs)
        #self._model.add(dense)
        #self._model.add(outputs)

        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        #optimizer.minimize()
        self._model.compile(optimizer=optimizer,
                            #loss=tf.losses.BinaryCrossentropy(),
                            metrics=[tf.metrics.BinaryAccuracy("accuracy")],)

    # Define a training method.
    #
    # Note that we need to use `raw_tf_function` (a faster variant of `tf.function`)
    # and manual `tf.GradientTape` for efficiency (using `fit` or `train_on_batch`
    # on extremely small batches has considerable overhead).
    @wrappers.raw_tf_function(dynamic_dims=1)
    def train(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:
        # TODO: Perform training, using the loss from the REINFORCE algorithm.
        # The easiest approach is to use the `sample_weight` argument of
        # the `__call__` method of a suitable subclass of `tf.losses.Loss`,
        # but you can also construct a loss instance with `reduction=tf.losses.Reduction.NONE`
        # and perform the weighting manually.
        #raise NotImplementedError()

        print("states:", states.shape)
        print("actions:", actions.shape)
        print("returns:", returns.shape)
        #print("states:", states)
        #print("actions:", actions)
        #print("returns:", returns)
        loss_subclass = tf.losses.BinaryCrossentropy()
        #loss_subclass = tf.losses.SparseCategoricalCrossentropy()
        # with for loop (better to vectorize it?)
        #loss = 0
        #for i in range(states.shape[0]):
        #    loss += loss_subclass(y_true=actions, y_pred=states, sample_weight=returns)

        with tf.GradientTape() as tape:
            #predict = self._model(states)
            predict = self._model.predict(states)
            #print("predict:", predict)
            acts = tf.transpose([actions, actions])
            loss = loss_subclass(y_true=acts, y_pred=predict, sample_weight=returns)

        #print("loss:", loss)
        loss = tf.reduce_mean(loss)
        #self._model.optimizer.minimize(loss=loss, tape=tape)
        var_list = self._model.trainable_variables
        #var_list = self._model.optimizer.variables()
        #print("var_list:", var_list)

        #grad = tape.gradient(loss, var_list)
        #var_list = self._model.trainable_weights
        print(loss)
        self._model.optimizer.minimize(loss=loss, var_list=var_list, tape=tape)

        #for variable, gradient in zip(var_list, grad):
        #    variable.assign_sub(args.learning_rate * gradient)



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
            print("state, done:", state, done)
            while not done:
                # TODO: Choose `action` according to probabilities
                # distribution (see `np.random.choice`), which you
                # can compute using `agent.predict` and current `state`.
                #action = ...
                prob_distr = agent.predict([state])
                print("prob_distr", prob_distr)
                #action = np.random.choice(a=1, p=prob_distr[0])
                action = np.random.choice(a=2, p=prob_distr[0])
                #action = np.random.choice(a=2, p=prob_distr)

                # Straka:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # TODO: Compute returns from the received rewards
            returns = []
            for i in range(0, len(rewards)):
                one_return = np.sum(rewards[:i])
                returns.append(one_return)

            # TODO: Add states, actions and returns to the training batch
            #batch_states.append(states)
            #batch_actions.append(actions)
            #batch_returns.append(returns)
            batch_actions += actions
            batch_states += states
            batch_returns += returns

        # TODO: Train using the generated batch.
        #agent.train(states=batch_states, actions=batch_actions, returns=batch_returns)
        agent.train(batch_states, batch_actions, batch_returns)

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO: Choose a greedy action
            #action = ...
            prob_distr = agent.predict(state)
            action = np.argmax(prob_distr)

            # Straka:
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("CartPole-v1"), args.seed, args.render_each)

    main(env, args)
