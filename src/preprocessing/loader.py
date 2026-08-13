import tensorflow as tf


def load_image(image_path, channels=1):
    """
    Load a PNG image from disk.
    """

    image = tf.io.read_file(image_path)

    image = tf.image.decode_png(
        image,
        channels=channels
    )

    return image