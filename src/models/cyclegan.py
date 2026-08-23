import tensorflow as tf
from keras import Model

from src.models.generator import build_generator
from src.models.discriminator import build_discriminator
from src.models.losses import (
    adversarial_loss,
    discriminator_loss,
    cycle_consistency_loss,
    identity_loss,
)


class CycleGAN(Model):

    def __init__(
        self,
        generator_g,
        generator_f,
        discriminator_x,
        discriminator_y,
        lambda_cycle=10.0,
        lambda_identity=5.0,
    ):
        super().__init__()

        self.generator_g = generator_g
        self.generator_f = generator_f

        self.discriminator_x = discriminator_x
        self.discriminator_y = discriminator_y

        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def compile(
        self,
        generator_g_optimizer,
        generator_f_optimizer,
        discriminator_x_optimizer,
        discriminator_y_optimizer,
    ):
        super().compile()

        self.generator_g_optimizer = generator_g_optimizer
        self.generator_f_optimizer = generator_f_optimizer

        self.discriminator_x_optimizer = discriminator_x_optimizer
        self.discriminator_y_optimizer = discriminator_y_optimizer

    def train_step(self, batch_data):

        real_x, real_y = batch_data

        with tf.GradientTape(persistent=True) as tape:

            # T1 → T2
            fake_y = self.generator_g(real_x, training=True)

            # T2 → T1
            fake_x = self.generator_f(real_y, training=True)

            # Cycle reconstruction
            reconstructed_x = self.generator_f(
                fake_y,
                training=True
            )

            reconstructed_y = self.generator_g(
                fake_x,
                training=True
            )

            # Identity mapping
            identity_x = self.generator_f(
                real_x,
                training=True
            )

            identity_y = self.generator_g(
                real_y,
                training=True
            )

            # Discriminator predictions
            disc_real_x = self.discriminator_x(
                real_x,
                training=True
            )

            disc_fake_x = self.discriminator_x(
                fake_x,
                training=True
            )

            disc_real_y = self.discriminator_y(
                real_y,
                training=True
            )

            disc_fake_y = self.discriminator_y(
                fake_y,
                training=True
            )

            # Generator adversarial losses
            generator_g_loss = adversarial_loss(
                disc_fake_y
            )

            generator_f_loss = adversarial_loss(
                disc_fake_x
            )

            # Cycle consistency
            cycle_x_loss = cycle_consistency_loss(
                real_x,
                reconstructed_x
            )

            cycle_y_loss = cycle_consistency_loss(
                real_y,
                reconstructed_y
            )

            total_cycle_loss = (
                cycle_x_loss + cycle_y_loss
            ) * self.lambda_cycle

            # Identity
            identity_x_loss = identity_loss(
                real_x,
                identity_x
            )

            identity_y_loss = identity_loss(
                real_y,
                identity_y
            )

            total_identity_loss = (
                identity_x_loss + identity_y_loss
            ) * self.lambda_identity

            # Total generator losses
            total_generator_g_loss = (
                generator_g_loss
                + total_cycle_loss
                + total_identity_loss
            )

            total_generator_f_loss = (
                generator_f_loss
                + total_cycle_loss
                + total_identity_loss
            )

            # Discriminator losses
            discriminator_x_loss = discriminator_loss(
                disc_real_x,
                disc_fake_x
            )

            discriminator_y_loss = discriminator_loss(
                disc_real_y,
                disc_fake_y
            )

        # Generator gradients
        generator_g_gradients = tape.gradient(
            total_generator_g_loss,
            self.generator_g.trainable_variables
        )

        generator_f_gradients = tape.gradient(
            total_generator_f_loss,
            self.generator_f.trainable_variables
        )

        # Discriminator gradients
        discriminator_x_gradients = tape.gradient(
            discriminator_x_loss,
            self.discriminator_x.trainable_variables
        )

        discriminator_y_gradients = tape.gradient(
            discriminator_y_loss,
            self.discriminator_y.trainable_variables
        )

        # Apply updates
        self.generator_g_optimizer.apply(
        generator_g_gradients,
        self.generator_g.trainable_variables
        )

        self.generator_f_optimizer.apply(
        generator_f_gradients,
        self.generator_f.trainable_variables
    )

        self.discriminator_x_optimizer.apply(
        discriminator_x_gradients,
        self.discriminator_x.trainable_variables
        )

        self.discriminator_y_optimizer.apply(
        discriminator_y_gradients,
        self.discriminator_y.trainable_variables
        )

        return {
            "generator_g_loss": total_generator_g_loss,
            "generator_f_loss": total_generator_f_loss,
            "discriminator_x_loss": discriminator_x_loss,
            "discriminator_y_loss": discriminator_y_loss,
            "cycle_loss": total_cycle_loss,
            "identity_loss": total_identity_loss,
        }





generator_g = build_generator()
generator_f = build_generator()

discriminator_x = build_discriminator()
discriminator_y = build_discriminator()