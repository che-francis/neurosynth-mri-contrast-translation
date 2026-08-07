import tensorflow as tf



def load_image(image_path):
    """
    Load a grayscale PNG image.

    Args:
        image_path (str): Path to image.

    Returns:
        tf.Tensor: Image tensor.
    """

    image = tf.io.read_file(image_path)

    image = tf.image.decode_png(
        image,
        channels=1
    )

    return image