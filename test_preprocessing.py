import os
import tensorflow as tf

from src.preprocessing.loader import load_image
from src.preprocessing.transforms import preprocess_image


image_path = "T1-T2-Dataset/Tr1/TrainT1/Image #10.png"

print("Current working directory:")
print(os.getcwd())

print("\nLooking for:")
print(image_path)

print("\nDoes file exist?")
print(os.path.exists(image_path))


if os.path.exists(image_path):

    image = load_image(image_path)
    processed_image = preprocess_image(image)

    print("\nOriginal image:")
    print("Shape:", image.shape)
    print("Data type:", image.dtype)

    print("\nProcessed image:")
    print("Shape:", processed_image.shape)
    print("Data type:", processed_image.dtype)

    print("\nPixel range:")
    print("Min:", tf.reduce_min(processed_image).numpy())
    print("Max:", tf.reduce_max(processed_image).numpy())

else:
    print("\nERROR: Python cannot find the image at this path.")