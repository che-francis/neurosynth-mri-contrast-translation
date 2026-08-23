import tensorflow as tf
from keras import layers


def downsample(filters, size, apply_batchnorm=True):
    """Downsampling block."""

    initializer = tf.random_normal_initializer(0., 0.02)

    block = tf.keras.Sequential()

    block.add(
        layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False
        )
    )

    if apply_batchnorm:
        block.add(layers.BatchNormalization())

    block.add(layers.LeakyReLU())

    return block


def build_discriminator():
    """Build a PatchGAN discriminator."""

    inputs = layers.Input(shape=[256, 256, 1])

    x = inputs

    x = downsample(64, 4, apply_batchnorm=False)(x)
    x = downsample(128, 4)(x)
    x = downsample(256, 4)(x)
    x = downsample(512, 4)(x)

    outputs = layers.Conv2D(
        1,
        4,
        strides=1,
        padding="same"
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)