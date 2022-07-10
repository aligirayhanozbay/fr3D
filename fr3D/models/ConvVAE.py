import tensorflow as tf

from .ConvAutoencoder import ConvAutoencoder, conv_block
from .ShallowDecoder import _get_original_shallow_decoder_layers

@tf.function
def sampling(z_mean: tf.Tensor, z_log_sigma: tf.Tensor):
    epsilon = tf.random.normal(tf.shape(z_log_sigma)[1:], mean=0.0, stddev=1.0)
    return z_mean + tf.exp(z_log_sigma/2)*epsilon

@tf.function
def SingleSampleVAELoss(x):
    z_mean, z_log_sigma = x
    return -0.5*tf.reduce_sum(1+z_log_sigma-z_mean**2-tf.exp(z_log_sigma))

@tf.function
def VAELoss(z_mean: tf.Tensor, z_log_sigma: tf.Tensor):
    losses_per_sample = tf.map_fn(SingleSampleVAELoss, (z_mean, z_log_sigma), fn_output_signature=z_mean.dtype)
    return tf.reduce_mean(losses_per_sample, axis=0)

class ConvVAE(ConvAutoencoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, auto_build=False, **kwargs)

        self.latent_sampler = self.create_latent_sampler(self.encoder.output_shape[1:])

        self.build(self.encoder.input_shape)
        

    @staticmethod
    def create_latent_sampler(latent_space_shape, bottleneck_size=None, kernel_size=3):
        '''
        if isinstance(latent_space_shape, int):
            latent_space_size = latent_space_shape
            latent_space_shape = (latent_space_shape,)
        else:
            latent_space_size = int(tf.reduce_prod(latent_space_shape))
        
        if bottleneck_size is None:
            bottleneck_size = latent_space_size

        inp = tf.keras.layers.Input(shape=latent_space_shape)
        if len(latent_space_shape)>1:
            x = tf.keras.layers.Flatten()(inp)
        else:
            x = inp

        dense_z_mean = tf.keras.layers.Dense(bottleneck_size, name='z_mean')(x)
        z_mean = tf.keras.layers.BatchNormalization()(dense_z_mean)
        
        dense_z_log_sigma = tf.keras.layers.Dense(bottleneck_size,kernel_initializer='zeros',name='z_log_sigma')(x)
        z_log_sigma = tf.keras.layers.BatchNormalization()(dense_z_log_sigma)

        z = tf.keras.layers.Lambda(lambda x: sampling(x[0],x[1]))([z_mean, z_log_sigma])
        z = tf.keras.layers.Dense(latent_space_size)(z)
        z = tf.keras.layers.Reshape(latent_space_shape)(z)

        latent_sampler = tf.keras.Model(inp, [z_mean, z_log_sigma, z], name='latent_sampler')

        return latent_sampler
        '''
        ndims = len(latent_space_shape)-1
        nfilters = latent_space_shape[-1]
        
        inp = tf.keras.layers.Input(shape=latent_space_shape)

        z_mean = conv_block(inp, nfilters, kernel_size, ndims, normalization='batchnorm')
        z_log_sigma = conv_block(inp, nfilters, kernel_size, ndims, normalization='batchnorm', kernel_initializer='zeros')
        z = tf.keras.layers.Lambda(lambda x: sampling(x[0],x[1]))([z_mean, z_log_sigma])
        z = conv_block(z, nfilters, kernel_size, ndims)

        latent_sampler = tf.keras.Model(inp, [z_mean, z_log_sigma, z], name='latent_sampler')

        return latent_sampler
        

    def call(self, inp, training=None):
        encoding = self.encoder(inp, training=training)
        z_mean, z_log_sigma, z = self.latent_sampler(encoding, training=training)
        reconstruction = self.decoder(z, training=training)

        if training:
            return z_mean, z_log_sigma, reconstruction
        else:
            return reconstruction

    def train_step(self, x):

        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            z_mean, z_log_sigma, y = self(x, training=True)
            vl = VAELoss(z_mean, z_log_sigma)
            rl = self.compute_loss(x, x, y)
            loss_val = rl + self._vae_loss_coeff*vl
        self.optimizer.minimize(loss_val, self.trainable_variables, tape=tape)

        return self.compute_metrics(x, x, y, sample_weight=None)
        

    def fit(self, *args, vae_loss_coeff=1e-3, **kwargs):
        self._vae_loss_coeff = vae_loss_coeff
        return super().fit(*args, **kwargs)
            

            
        
        
