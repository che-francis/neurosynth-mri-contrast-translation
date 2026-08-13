from pathlib import Path

import tensorflow as tf

from .loader import load_image
from .transforms import preprocess_image


BATCH_SIZE = 1
SHUFFLE_BUFFER_SIZE = 100


def get_image_paths(image_dir):
    """
    Get all PNG image paths from a directory.
    """

    return [
        str(path)
        for path in Path(image_dir).glob("*.png")
    ]


def process_image(image_path):
    """
    Load and preprocess a single image.
    """

    image = load_image(image_path)
    image = preprocess_image(image)

    return image


def create_dataset(image_dir):
    """
    Create a TensorFlow dataset from an image directory.
    """

    image_paths = get_image_paths(image_dir)

    dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    dataset = dataset.shuffle(
        buffer_size=min(
            SHUFFLE_BUFFER_SIZE,
            len(image_paths)
        )
    )

    dataset = dataset.map(
        process_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset