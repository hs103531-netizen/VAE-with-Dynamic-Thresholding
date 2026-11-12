"""Convolutional VAE (Keras 3 / TensorFlow backend).

원본 Colab 노트북의 인코더/디코더 구조를 그대로 옮기되, 입력 크기를 인자로 받도록 일반화했다.
(img_size=256 이면 노트북과 동일한 구조)
"""
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import tensorflow as tf
from keras import layers


@keras.saving.register_keras_serializable(package="vae_anomaly")
class Sampling(layers.Layer):
    """(z_mean, z_log_var)로부터 reparameterization trick으로 z를 샘플링."""

    def __init__(self, seed=1337, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = keras.random.normal(tf.shape(z_mean), seed=self.seed_generator)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def get_config(self):
        return {**super().get_config(), "seed": self.seed}


def build_encoder(img_size=256, latent_dim=2):
    inputs = keras.Input(shape=(img_size, img_size, 1))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")


def build_decoder(img_size=256, latent_dim=2):
    if img_size % 4 != 0:
        raise ValueError("img_size는 4의 배수여야 합니다 (stride-2 업샘플링 2회).")
    s = img_size // 4  # 256 -> 64 (노트북의 Dense(64*64*64)와 동일)
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(s * s * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((s, s, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    return keras.Model(latent_inputs, outputs, name="decoder")


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def call(self, inputs, training=False):
        _, _, z = self.encoder(inputs, training=training)
        return self.decoder(z, training=training)

    def compute_vae_losses(self, data, training):
        z_mean, z_log_var, z = self.encoder(data, training=training)
        reconstruction = self.decoder(z, training=training)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        return reconstruction_loss, kl_loss, reconstruction_loss + kl_loss

    def _update_trackers(self, rec, kl, total):
        self.total_loss_tracker.update_state(total)
        self.reconstruction_loss_tracker.update_state(rec)
        self.kl_loss_tracker.update_state(kl)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            rec, kl, total = self.compute_vae_losses(data, training=True)
        grads = tape.gradient(total, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return self._update_trackers(rec, kl, total)

    def test_step(self, data):
        # validation_split / validation_data 를 쓸 수 있도록 추가 (노트북에는 없었음)
        if isinstance(data, tuple):
            data = data[0]
        rec, kl, total = self.compute_vae_losses(data, training=False)
        return self._update_trackers(rec, kl, total)


def build_vae(img_size=256, latent_dim=2):
    vae = VAE(build_encoder(img_size, latent_dim), build_decoder(img_size, latent_dim))
    vae(tf.zeros((1, img_size, img_size, 1)))  # 가중치 생성 (load_weights 전에 필요)
    return vae
