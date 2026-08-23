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


def upsample(filters, size, apply_dropout=False):
    """Upsampling block."""

    initializer = tf.random_normal_initializer(0., 0.02)

    block = tf.keras.Sequential()

    block.add(
        layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False
        )
    )

    block.add(layers.BatchNormalization())

    if apply_dropout:
        block.add(layers.Dropout(0.5))

    block.add(layers.ReLU())

    return block


def build_generator():
    """Build the U-Net generator."""

    inputs = layers.Input(shape=[256, 256, 1])

    # Encoder
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
    ]

    # Decoder
    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]

    x = inputs
    skips = []

    # Encoder
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Decoder + skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    # Final output layer
    initializer = tf.random_normal_initializer(0., 0.02)

    outputs = layers.Conv2DTranspose(
        1,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh"
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)