import tensorflow as tf


IMG_SIZE = (256, 256)


def resize_image(image):
    """
    Resize image to 256 x 256.
    """

    return tf.image.resize(image, IMG_SIZE)


def normalize_image(image):
    """
    Convert pixel values from [0, 255] to [-1, 1].
    """

    image = tf.cast(image, tf.float32)

    image = (image / 127.5) - 1.0

    return image


def preprocess_image(image):
    """
    Resize and normalize an image.
    """

    image = resize_image(image)
    image = normalize_image(image)

    return image