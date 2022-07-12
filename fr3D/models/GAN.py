import tensorflow as tf
import copy
from .ConvAutoencoder import encoder_block

@tf.function
def per_sample_mse(y_true, y_pred):
    rank = tf.rank(y_true)
    return tf.reduce_mean((y_true-y_pred)**2, axis=tf.range(1,rank))

class GAN(tf.keras.models.Model):
    noise_mag = 0.05
    recons_loss_wt = 0.1
    def __init__(self, generator, generator_output_shape=None, discriminator_type='convolutional', discriminator_args = None):
        super().__init__()
        
        if generator_output_shape is None:
            generator_output_shape = generator.output_shape[1:]
        self.generator = generator

        if discriminator_args is None:
            discriminator_args = {}
        self.discriminator = self.discriminator_types[discriminator_type](generator_output_shape, **discriminator_args)

    @property
    def discriminator_types(self):
        dt = {'convolutional': self.make_convolutional_discriminator}
        return dt

    @staticmethod
    def make_convolutional_discriminator(input_shape, **discriminator_args):

        discriminator_args = copy.deepcopy(discriminator_args)
        
        #automatically pick required missing discriminator settings
        if 'activation' not in discriminator_args:
            discriminator_args['activation'] = tf.nn.leaky_relu
        elif discriminator_args['activation'] == 'leaky_relu':
            discriminator_args['activation'] = tf.nn.leaky_relu

        if 'blocks_per_level' not in discriminator_args:
            discriminator_args['blocks_per_level'] = 2

        if 'pool_size' not in discriminator_args:
            discriminator_args['pool_size'] = 4

        if 'pool_type' not in discriminator_args:
            discriminator_args['pool_type'] = 'max'

        if 'levels' not in discriminator_args:
            spatial_sizes = input_shape[:-1] if tf.keras.backend.image_data_format()=='channels_last' else input_shape[1:]
            min_spatial_size = min(spatial_sizes)
            discriminator_args['levels'] = max(2,int(tf.floor(tf.math.log(float(min_spatial_size))/tf.math.log(float(discriminator_args['pool_size']))))-1)

        if 'base_filters' not in discriminator_args:
            discriminator_args['base_filters'] = max(2, 8192/(2**(discriminator_args['levels']+6)))
            
        
        inp = tf.keras.layers.Input(shape=input_shape)
        x = encoder_block(inp, **discriminator_args)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        return tf.keras.Model(inp, x, name='conv_discriminator')

    def compile(self, d_optimizer, g_optimizer,
                discriminator_loss = None,
                global_batch_size: int = None):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.reconstruction_loss = per_sample_mse
        self.discriminator_loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.global_batch_size = global_batch_size

    def test_step(self, x):
        sensor_inputs, full_fields = x
        pred = self.generator(x)
        loss_val = self.reconstruction_loss(full_fields, pred)
        loss_val = tf.reduce_sum(loss_val)/self.global_batch_size
        
        return {'reconstruction_loss': loss_val}

class ViTGAN(GAN):
    def train_step(self, x):

        sensor_inputs, full_fields = x
        batch_size = tf.shape(full_fields)[0]

        global_batch_size = self.global_batch_size or batch_size

        # Train the generator
        misleading_labels = tf.zeros((batch_size,1))
        with tf.GradientTape() as tape:
            tape.watch(self.generator.trainable_variables)
            generated_images = self.generator(x)
            predictions = self.discriminator(generated_images)
            if self.reconstruction_loss is not None:
                recons_loss = self.reconstruction_loss(full_fields, generated_images)
                recons_loss = tf.reduce_sum(recons_loss)/self.global_batch_size
            else:
                recons_loss = 0.0
            disc_loss = self.discriminator_loss(misleading_labels, predictions)
            disc_loss = tf.reduce_sum(disc_loss)/self.global_batch_size
            g_loss =  disc_loss + self.recons_loss_wt*recons_loss
        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        combined_images = tf.concat([generated_images, full_fields], axis=0)

        # Assemble labels discriminating real from fake images
        # Add random noise to the labels - important trick!
        labels = tf.concat(
            [tf.ones((batch_size, 1))-self.noise_mag*tf.random.uniform((batch_size,1)),
             tf.zeros((batch_size, 1))+self.noise_mag*tf.random.uniform((batch_size,1))],
            axis=0)

        # Train the discriminator
        with tf.GradientTape() as tape:
            tape.watch(self.discriminator.trainable_variables)
            predictions = self.discriminator(combined_images)
            d_loss = self.discriminator_loss(labels, predictions)
            d_loss = tf.reduce_sum(d_loss)/self.global_batch_size
        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        
        return {"g_loss": g_loss, "d_loss": d_loss, "reconstruction_loss": recons_loss}

from .ConvVAE import VAELoss
class VAEGAN(GAN):
    def train_step(self, x):

        full_fields = x

        batch_size = tf.shape(full_fields)[0]

        global_batch_size = self.global_batch_size or batch_size

        # Train the generator
        misleading_labels = tf.zeros((batch_size,1))
        with tf.GradientTape() as tape:
            tape.watch(self.generator.trainable_variables)
            z_mean, z_log_sigma, generated_images = self.generator(x, training=True)
            predictions = self.discriminator(generated_images)

            elbo_loss = VAELoss(z_mean, z_log_sigma)

            recons_loss = self.reconstruction_loss(full_fields, generated_images)
            recons_loss = tf.reduce_sum(recons_loss)/self.global_batch_size

            disc_loss = self.discriminator_loss(misleading_labels, predictions)
            disc_loss = tf.reduce_sum(disc_loss)/self.global_batch_size

            g_loss =  disc_loss + self.recons_loss_wt*recons_loss + self.generator._vae_loss_coeff*elbo_loss
        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        
        combined_images = tf.concat([generated_images, full_fields], axis=0)

        # Assemble labels discriminating real from fake images
        # Add random noise to the labels - important trick!
        labels = tf.concat(
            [tf.ones((batch_size, 1))-self.noise_mag*tf.random.uniform((batch_size,1)),
             tf.zeros((batch_size, 1))+self.noise_mag*tf.random.uniform((batch_size,1))],
            axis=0)

        # Train the discriminator
        with tf.GradientTape() as tape:
            tape.watch(self.discriminator.trainable_variables)
            predictions = self.discriminator(combined_images)
            d_loss = self.discriminator_loss(labels, predictions)
            d_loss = tf.reduce_sum(d_loss)/self.global_batch_size
        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        
        return {"g_loss": g_loss, "d_loss": d_loss, "reconstruction_loss": recons_loss, "elbo_loss": elbo_loss}

    def test_step(self, x):
        full_fields = x
        pred = self.generator(x)
        loss_val = self.reconstruction_loss(full_fields, pred)
        loss_val = tf.reduce_sum(loss_val)/self.global_batch_size
        
        return {'reconstruction_loss': loss_val}

if __name__ == '__main__':

    m = GAN.make_convolutional_discriminator((64,64,64,3))
    m.summary()
    
