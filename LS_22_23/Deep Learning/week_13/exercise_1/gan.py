#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import Dict
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tf.get_logger().addFilter(lambda m: "Analyzer.lamba_check" not in m.getMessage())  # Avoid pesky warning

from mnist import MNIST

# 3d76595a-e687-11e9-9ce9-00505601122b
# 594215cf-e687-11e9-9ce9-00505601122b

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--dataset", default="mnist", type=str, help="MNIST-like dataset to use.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--discriminator_layers", default=[128], type=int, nargs="+", help="Discriminator layers.")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
parser.add_argument("--generator_layers", default=[128], type=int, nargs="+", help="Generator layers.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--train_size", default=None, type=int, help="Limit on the train set size.")
parser.add_argument("--z_dim", default=100, type=int, help="Dimension of Z.")
# If you add more arguments, ReCodEx will keep them with your default values.


# The GAN model
class GAN(tf.keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()

        self._seed = args.seed
        self._z_dim = args.z_dim
        self._z_prior = tfp.distributions.Normal(tf.zeros(args.z_dim), tf.ones(args.z_dim))

        # TODO: Define `self.generator` as a Model, which
        # - takes vectors of [args.z_dim] shape on input
        # - applies len(args.generator_layers) dense layers with ReLU activation,
        #   i-th layer with args.generator_layers[i] units
        # - applies output dense layer with MNIST.H * MNIST.W * MNIST.C units
        #   and a suitable output activation
        # - reshapes the output (tf.keras.layers.Reshape) to [MNIST.H, MNIST.W, MNIST.C]

        generator = tf.keras.Sequential([
            tf.keras.layers.Input(shape=args.z_dim),
        ])
        for i in range(len(args.generator_layers)):
            generator.add(tf.keras.layers.Dense(units=args.generator_layers[i], activation=tf.nn.relu))
        generator.add(tf.keras.layers.Dense(units=MNIST.H * MNIST.W * MNIST.C, activation=tf.nn.sigmoid))
        generator.add(tf.keras.layers.Reshape([MNIST.H, MNIST.W, MNIST.C]))
        self.generator = generator

        # TODO: Define `self.discriminator` as a Model, which
        # - takes input images with shape [MNIST.H, MNIST.W, MNIST.C]
        # - flattens them
        # - applies len(args.discriminator_layers) dense layers with ReLU activation,
        #   i-th layer with args.discriminator_layers[i] units
        # - applies output dense layer with one output and a suitable activation function

        discriminator = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C]),
            tf.keras.layers.Flatten(),
        ])
        for i in range(len(args.discriminator_layers)):
            discriminator.add(tf.keras.layers.Dense(units=args.discriminator_layers[i], activation=tf.nn.relu))
        discriminator.add(tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid))
        self.discriminator = discriminator

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    # We override `compile`, because we want to use two optimizers.
    def compile(
        self, discriminator_optimizer: tf.optimizers.Optimizer, generator_optimizer: tf.optimizers.Optimizer
    ) -> None:
        super().compile(
            loss=tf.losses.BinaryCrossentropy(),
            metrics=tf.metrics.BinaryAccuracy("discriminator_accuracy"),
        )

        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer

    def train_step(self, images: tf.Tensor) -> Dict[str, tf.Tensor]:
        # TODO: Generator training. With a Gradient tape:
        # - generate as many random latent samples as there are `images`, by a single call
        #   to `self._z_prior.sample`; also pass `seed=self._seed` for replicability;
        # - pass the samples through a generator; do not forget about `training=True`
        # - run discriminator on the generated images, also using `training=True` (even if
        #   not updating discriminator parameters, we want to perform possible BatchNorm in it)
        # - compute `generator_loss` using `self.compiled_loss`, with ones as target labels
        #   (`tf.ones_like` might come handy).
        # Then, run an optimizer step with respect to generator trainable variables.
        # Do not forget that we created generator_optimizer in the `compile` override.
        with tf.GradientTape() as tape:
            number_of_samples = tf.shape(images)[0]
            latent_samples = self._z_prior.sample(sample_shape=number_of_samples, seed=self._seed)
            generated = self.generator(latent_samples, training=True)
            discriminated = self.discriminator(generated, training=True)

            ones = tf.ones_like(discriminated)
            generator_loss = self.compiled_loss(ones, discriminated)

            self.generator_optimizer.minimize(generator_loss, self.generator.trainable_variables, tape=tape)


        # TODO: Discriminator training. Using a Gradient tape:
        # - discriminate `images` with `training=True`, storing
        #   results in `discriminated_real`
        # - discriminate images generated in generator training with `training=True`,
        #   storing results in `discriminated_fake`
        # - compute `discriminator_loss` by summing
        #   - `self.compiled_loss` on `discriminated_real` with suitable targets,
        #   - `self.compiled_loss` on `discriminated_fake` with suitable targets.
        # Then, run an optimizer step with respect to discriminator trainable variables.
        # Do not forget that we created discriminator_optimizer in the `compile` override.
        with tf.GradientTape() as tape:
            discriminated_real = self.discriminator(images, training=True)
            discriminated_fake = self.discriminator(generated, training=True)

            ones = tf.ones_like(discriminated_real)
            zeros = tf.zeros_like(discriminated_fake)
            loss_on_real = self.compiled_loss(ones, discriminated_real)
            loss_on_fake = self.compiled_loss(zeros, discriminated_fake)
            discriminator_loss = loss_on_real + loss_on_fake

            self.discriminator_optimizer.minimize(discriminator_loss, self.discriminator.trainable_variables, tape=tape)

        # TODO: Update the discriminator accuracy metric -- call the
        # `self.compiled_metrics.update_state` twice, with the same arguments
        # the `self.compiled_loss` was called during discriminator loss computation.
        self.compiled_metrics.update_state(ones, discriminated_real)
        self.compiled_metrics.update_state(zeros, discriminated_fake)

        return {
            "discriminator_loss": discriminator_loss,
            "generator_loss": generator_loss,
            **{metric.name: metric.result() for metric in self.metrics},
        }

    def generate(self, epoch: int, logs: Dict[str, tf.Tensor]) -> None:
        GRID = 20

        # Generate GRIDxGRID images
        random_images = self.generator(self._z_prior.sample(GRID * GRID, seed=self._seed), training=False)

        # Generate GRIDxGRID interpolated images
        if self._z_dim == 2:
            # Use 2D grid for sampled Z
            starts = tf.stack([-2 * tf.ones(GRID), tf.linspace(-2., 2., GRID)], -1)
            ends = tf.stack([2 * tf.ones(GRID), tf.linspace(-2., 2., GRID)], -1)
        else:
            # Generate random Z
            starts = self._z_prior.sample(GRID, seed=self._seed)
            ends = self._z_prior.sample(GRID, seed=self._seed)
        interpolated_z = tf.concat(
            [starts[i] + (ends[i] - starts[i]) * tf.linspace(0., 1., GRID)[:, tf.newaxis] for i in range(GRID)],
            axis=0)
        interpolated_images = self.generator(interpolated_z, training=False)

        # Stack the random images, then an empty row, and finally interpolated images
        image = tf.concat([
            tf.concat([tf.concat(list(images), axis=1) for images in tf.split(random_images, GRID)], axis=0),
            tf.zeros([MNIST.H * GRID, MNIST.W, MNIST.C]),
            tf.concat([tf.concat(list(images), axis=1) for images in tf.split(interpolated_images, GRID)], axis=0),
        ], axis=1)
        with self.tb_callback._train_writer.as_default(step=epoch):
            tf.summary.image("images", image[tf.newaxis])


def main(args: argparse.Namespace) -> Dict[str, float]:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        # tf.data.experimental.enable_debug_mode()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load data
    mnist = MNIST(args.dataset, size={"train": args.train_size})
    train = mnist.train.dataset.map(lambda example: example["images"])
    train = train.shuffle(mnist.train.size, args.seed)
    train = train.batch(args.batch_size)

    # Create the network and train
    network = GAN(args)
    network.compile(
        discriminator_optimizer=tf.optimizers.Adam(jit_compile=False),
        generator_optimizer=tf.optimizers.Adam(jit_compile=False),
    )
    logs = network.fit(train, epochs=args.epochs, callbacks=[
        tf.keras.callbacks.LambdaCallback(on_epoch_end=network.generate), network.tb_callback])

    # Return the loss and the discriminator accuracy for ReCodEx to validate.
    return {metric: logs.history[metric][-1] for metric in ["loss", "discriminator_accuracy"]}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
