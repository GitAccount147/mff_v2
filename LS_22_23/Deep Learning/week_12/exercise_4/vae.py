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
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.") #50
parser.add_argument("--dataset", default="mnist", type=str, help="MNIST-like dataset to use.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.") #False
parser.add_argument("--decoder_layers", default=[500, 500], type=int, nargs="+", help="Decoder layers.")
parser.add_argument("--encoder_layers", default=[500, 500], type=int, nargs="+", help="Encoder layers.")
parser.add_argument("--epochs", default=3, type=int, help="Number of epochs.") #50
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--train_size", default=500, type=int, help="Limit on the train set size.") #None
parser.add_argument("--z_dim", default=100, type=int, help="Dimension of Z.") #100
# If you add more arguments, ReCodEx will keep them with your default values.


# The VAE model
class VAE(tf.keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()

        self._seed = args.seed
        self._z_dim = args.z_dim
        self._z_prior = tfp.distributions.Normal(tf.zeros(args.z_dim), tf.ones(args.z_dim))

        # TODO: Define `self.encoder` as a `tf.keras.Model`, which
        # - takes input images with shape `[MNIST.H, MNIST.W, MNIST.C]`
        # - flattens them
        # - applies `len(args.encoder_layers)` dense layers with ReLU activation,
        #   i-th layer with `args.encoder_layers[i]` units
        # - generates two outputs `z_mean` and `z_sd`, each passing the result
        #   of the above bullet through its own dense layer of `args.z_dim` units,
        #   with `z_sd` using exponential function as activation to keep it positive.
        encoder_inputs = tf.keras.layers.Input([MNIST.H, MNIST.W, MNIST.C])
        enc_model = tf.keras.layers.Flatten()(encoder_inputs)
        for i in range(len(args.encoder_layers)):
            enc_model = tf.keras.layers.Dense(args.encoder_layers[i], activation=tf.keras.activations.relu)(enc_model)
        z_mean = tf.keras.layers.Dense(args.z_dim)(enc_model) # activation?
        z_sd = tf.keras.layers.Dense(args.z_dim, activation=tf.keras.activations.exponential)(enc_model)

        self.encoder = tf.keras.Model(inputs=encoder_inputs, outputs=[z_mean, z_sd])

        # TODO: Define `self.decoder` as a `tf.keras.Model`, which
        # - takes vectors of `[args.z_dim]` shape on input
        # - applies `len(args.decoder_layers)` dense layers with ReLU activation,
        #   i-th layer with `args.decoder_layers[i]` units
        # - applies output dense layer with `MNIST.H * MNIST.W * MNIST.C` units
        #   and a suitable output activation (na cviku rika sigmoid)
        # - reshapes the output (`tf.keras.layers.Reshape`) to `[MNIST.H, MNIST.W, MNIST.C]`
        decoder_inputs = tf.keras.layers.Input([args.z_dim])
        dec_model = decoder_inputs
        for i in range(len(args.decoder_layers)):
            dec_model = tf.keras.layers.Dense(units=args.decoder_layers[i], activation=tf.keras.activations.relu)(dec_model)
        dec_model = tf.keras.layers.Dense(units=MNIST.H * MNIST.W * MNIST.C, activation=tf.keras.activations.sigmoid)(dec_model)
        dec_model = tf.keras.layers.Reshape([MNIST.H, MNIST.W, MNIST.C])(dec_model)
        self.decoder = tf.keras.Model(inputs=decoder_inputs, outputs=dec_model)

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    def train_step(self, images: tf.Tensor) -> Dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            # TODO: Compute `z_mean` and `z_sd` of the given images using `self.encoder`.
            # Note that you should pass `training=True` to the `self.encoder`.
            z_mean, z_sd = self.encoder(images, training=True)

            # TODO: Sample `z` from a Normal distribution with mean `z_mean` and
            # standard deviation `z_sd`. Start by creating corresponding
            # distribution `tfp.distributions.Normal(...)` and then run the
            # `sample(seed=self._seed)` method.
            #
            # Note that the distributions in `tfp` are already reparametrized if possible,
            # so you do not need to implement the reparametrization trick manually.
            # For a given distribution, you can use the `reparameterization_type` member
            # to check if it is reparametrized or not.

            dist = tfp.distributions.Normal(loc=z_mean, scale=z_sd)
            z = dist.sample(seed=self._seed)


            # TODO: Decode images using `z` (also passing `training=True` to the `self.decoder`).
            dec_imgs = self.decoder(z, training=True)


            # TODO: Compute `reconstruction_loss` using the `self.compiled_loss`.
            reconstruction_loss = self.compiled_loss(images, dec_imgs)


            # TODO: Compute `latent_loss` as a mean of KL divergences of suitable distributions.
            # Note that the `tfp` distributions offer a method `kl_divergence`.
            latent_loss = dist.kl_divergence(tfp.distributions.Normal(loc=tf.zeros(self._z_dim), scale=tf.ones(self._z_dim)))
            latent_loss = tf.math.reduce_mean(latent_loss)


            # TODO: Compute `loss` as a sum of the `reconstruction_loss` (multiplied by the number
            # of pixels in an image) and the `latent_loss` (multiplied by self._z_dim).
            loss = (MNIST.H * MNIST.W)*reconstruction_loss + self._z_dim*latent_loss


        # TODO: Perform a single optimizer step, with respect to trainable variables
        # of both the encoder and the decoder.

        self.optimizer.minimize(loss, self.encoder.trainable_variables+self.decoder.trainable_variables, tape=tape)

        return {"reconstruction_loss": reconstruction_loss, "latent_loss": latent_loss, "loss": loss}

    def generate(self, epoch: int, logs: Dict[str, tf.Tensor]) -> None:
        GRID = 20

        # Generate GRIDxGRID images
        random_images = self.decoder(self._z_prior.sample(GRID * GRID, seed=self._seed), training=False)

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
        interpolated_images = self.decoder(interpolated_z, training=False)

        # Stack the random images, then an empty row, and finally interpolated images
        image = tf.concat([
            tf.concat([tf.concat(list(images), axis=1) for images in tf.split(random_images, GRID)], axis=0),
            tf.zeros([MNIST.H * GRID, MNIST.W, MNIST.C]),
            tf.concat([tf.concat(list(images), axis=1) for images in tf.split(interpolated_images, GRID)], axis=0),
        ], axis=1)
        with self.tb_callback._train_writer.as_default(step=epoch):
            tf.summary.image("images", image[tf.newaxis])


def main(args: argparse.Namespace) -> float:
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
    network = VAE(args)
    network.compile(optimizer=tf.optimizers.Adam(jit_compile=False), loss=tf.losses.BinaryCrossentropy())
    logs = network.fit(train, epochs=args.epochs, callbacks=[
        tf.keras.callbacks.LambdaCallback(on_epoch_end=network.generate), network.tb_callback])

    # Return loss for ReCodEx to validate
    return logs.history["loss"][-1]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
