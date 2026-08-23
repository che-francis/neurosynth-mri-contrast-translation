import tensorflow as tf


loss_object = tf.keras.losses.BinaryCrossentropy(
    from_logits=True
)


def adversarial_loss(discriminator_output):
    """
    Adversarial loss for generated images.
    """

    return loss_object(
        tf.ones_like(discriminator_output),
        discriminator_output
    )


def discriminator_loss(real_output, generated_output):
    """
    Discriminator loss for real and generated images.
    """

    real_loss = loss_object(
        tf.ones_like(real_output),
        real_output
    )

    generated_loss = loss_object(
        tf.zeros_like(generated_output),
        generated_output
    )

    return real_loss + generated_loss


def cycle_consistency_loss(real_image, reconstructed_image):
    """
    Cycle consistency loss.
    """

    return tf.reduce_mean(
        tf.abs(real_image - reconstructed_image)
    )


def identity_loss(real_image, same_image):
    """
    Identity mapping loss.
    """

    return tf.reduce_mean(
        tf.abs(real_image - same_image)
    )