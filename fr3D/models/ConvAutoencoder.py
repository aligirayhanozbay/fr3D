import tensorflow as tf
import os

conv_layers = {1: tf.keras.layers.Conv1D,
               2:tf.keras.layers.Conv2D,
               3:tf.keras.layers.Conv3D}
deconv_layers = {1: tf.keras.layers.Conv1DTranspose,
                 2:tf.keras.layers.Conv2DTranspose,
                 3:tf.keras.layers.Conv3DTranspose}
average_pool_layers = {1: tf.keras.layers.AveragePooling1D,
                       2: tf.keras.layers.AveragePooling2D,
                       3: tf.keras.layers.AveragePooling3D}
max_pool_layers = {1: tf.keras.layers.MaxPooling1D,
                   2: tf.keras.layers.MaxPooling2D,
                   3: tf.keras.layers.MaxPooling3D}
normalization_layers = {'layernorm': tf.keras.layers.LayerNormalization,
                        'batchnorm': tf.keras.layers.BatchNormalization}

def conv_block(input_tensor, filters: int, kernel_size: int = 3, ndims: int = 3, activation = None, normalization = None, pre_activation: bool = False, residual_connection: bool = False, kernel_initializer=None, dropout=False):

    x = input_tensor

    if dropout:
        x = tf.keras.layers.Dropout(dropout)(x)

    for k in range(2):
        if normalization is not None:
            norm_axis = 1 if tf.keras.backend.image_data_format()=='channels_first' else -1
            x = normalization_layers[normalization](axis=norm_axis)(x)
        if pre_activation:
            x = tf.keras.layers.Activation(tf.nn.relu)(x)
        x = conv_layers[ndims](filters,
                               kernel_size,
                               padding='same',
                               activation=activation,
                               kernel_initializer=kernel_initializer)(x)

    if residual_connection:
        x = x + input_tensor
        
    return x

def encoder_block(input_tensor, levels: int, base_filters: int, kernel_size: int = 3, activation = None, normalization = None, pre_activation: bool = True, residual_connection: bool = True, blocks_per_level: int = 4, pool_size: int = 2, dropout=False, pool_type='average'):

    if pool_type == 'average':
        pool_layers = average_pool_layers
    elif pool_type == 'max':
        pool_layers = max_pool_layers
    else:
        raise(ValueError(f'Unrecognized pooling type {pool_type} - choose average or max'))

    ndims = len(input_tensor.shape)-2

    x = input_tensor

    x = conv_layers[ndims](base_filters, 7, padding='same', activation=activation)(x)
    x = pool_layers[ndims](padding='same', pool_size = pool_size)(x)

    for k in range(levels):
        level_filters = (2**(k+1))*base_filters
        x = conv_layers[ndims](level_filters, kernel_size, activation=activation, padding='same')(x)
        for _ in range(blocks_per_level):
            x = conv_block(x,
                           filters=level_filters,
                           kernel_size=kernel_size,
                           ndims=ndims,
                           activation=activation,
                           normalization=normalization,
                           pre_activation=pre_activation,
                           residual_connection=residual_connection,
                           dropout=dropout)

        if k < (levels-1):
            x = pool_layers[ndims](padding='same', pool_size = pool_size)(x)

    return x

def decoder_block(input_tensor, levels: int, output_filters: int, kernel_size: int = 3, activation = None, final_activation = None, pre_activation: bool = True, residual_connection: bool = True, normalization = None, blocks_per_level: int = 4, deconv_stride: int = 2, dropout=False):

    if isinstance(input_tensor, tuple) or isinstance(input_tensor, list):
        make_model=True
        input_tensor = tf.keras.layers.Input(shape=input_tensor)
    else:
        make_model=False

    norm_axis = 1 if tf.keras.backend.image_data_format()=='channels_first' else -1
    base_filters = input_tensor.shape[norm_axis]
    ndims = len(input_tensor.shape)-2
    
    x = input_tensor

    for _ in range(blocks_per_level):
        x = conv_block(x,
                       filters=base_filters,
                       kernel_size=kernel_size,
                       ndims=ndims,
                       activation=activation,
                       normalization=normalization,
                       pre_activation=pre_activation,
                       residual_connection=residual_connection,
                       dropout=dropout)

    for k in range(levels):
        level_filters = base_filters//(2**(k+1))
        x = conv_layers[ndims](level_filters, kernel_size, activation=activation, padding='same')(x)
        x = deconv_layers[ndims](level_filters, kernel_size, strides=deconv_stride, activation=activation, padding='same')(x)
        for _ in range(blocks_per_level):
            x = conv_block(x,
                           filters=level_filters,
                           kernel_size=kernel_size,
                           ndims=ndims,
                           activation=activation,
                           normalization=normalization,
                           pre_activation=pre_activation,
                           residual_connection=residual_connection,
                           dropout=dropout)

    x = conv_layers[ndims](output_filters, kernel_size, activation=final_activation, padding='same')(x)

    if make_model:
        x = tf.keras.Model(input_tensor, x, name='decoder_block')
            
    return x


class ConvAutoencoder(tf.keras.models.Model):
    def __init__(self, levels: int, base_filters: int,
                 kernel_size: int = 3,
                 activation = None,
                 normalization: str = None,
                 pre_activation: bool = True,
                 residual_connection: bool = True,
                 blocks_per_level: int = 4,
                 pool_size: int = 2,
                 dropout=False,
                 input_shape = None, ndims: int = 3, filters: int = None, auto_build=True):

        super().__init__()

        if activation == 'leaky_relu':
            activation = tf.nn.leaky_relu
        
        if input_shape is None and filters is None:
            raise(ValueError('Either input_shape or (filters and ndims) must be provided.'))
        elif input_shape is None:
            if tf.keras.backend.image_data_format() == 'channels_last':
                input_shape = [None for _ in range(ndims)] + [filters]
            else:
                input_shape = [filters] + [None for _ in range(ndims)]
        
        inp = tf.keras.layers.Input(input_shape)
        channel_axis = -1 if tf.keras.backend.image_data_format() == 'channels_last' else 1
        output_filters = inp.shape[channel_axis]

        encoder_out = self.create_encoder(input_tensor=inp,
                                          levels=levels,
                                          base_filters=base_filters,
                                          kernel_size=kernel_size,
                                          activation=activation,
                                          normalization=normalization,
                                          pre_activation=pre_activation,
                                          residual_connection=residual_connection,
                                          blocks_per_level=blocks_per_level,
                                          pool_size=pool_size,
                                          dropout=dropout)

        self.encoder = tf.keras.Model(inp, encoder_out, name='encoder')

        decoder_out = self.create_decoder(input_tensor=encoder_out,
                                          output_filters=output_filters,
                                          levels=levels,
                                          kernel_size=kernel_size,
                                          activation=activation,
                                          normalization=normalization,
                                          pre_activation=pre_activation,
                                          residual_connection=residual_connection,
                                          blocks_per_level=blocks_per_level,
                                          deconv_stride=pool_size,
                                          dropout=dropout)

        self.decoder = tf.keras.Model(encoder_out, decoder_out, name='decoder')

        if auto_build:
            self.build(inp.shape)

    @staticmethod
    def create_encoder(*args, **kwargs):
        return encoder_block(*args, **kwargs)

    @staticmethod
    def create_decoder(*args, **kwargs):
        return decoder_block(*args, **kwargs)

    @property
    def output_shape(self):
        return self.decoder.output_shape
    
    def call(self, inp, training=None):
        return self.decoder(self.encoder(inp, training=training), training=training)
                     
    def train_step(self, x):
        return super().train_step((x,x))

    def test_step(self, x):
        return super().test_step((x,x))

class ConvAutoencoderC(ConvAutoencoder):

    def __init__(self, dense_input_units, *args, autoencoder_input_shape=None, auto_build=True, **kwargs):
        super().__init__(*args, input_shape=autoencoder_input_shape, auto_build=False, **kwargs)

        self.latent_space_embedder = self.make_latent_space_embedder(self.decoder.input_shape[1:], dense_input_units)

        if auto_build:
            self.build(self.encoder.input_shape)

    @staticmethod
    def make_latent_space_embedder(latent_space_shape, input_units, dense_activation=tf.nn.leaky_relu):
        
        ndims = len(latent_space_shape)-1
        nfilters = latent_space_shape[0 if tf.keras.backend.image_data_format() == 'channels_first' else -1]

        inp = tf.keras.layers.Input(shape=(input_units,))
        x = tf.keras.layers.Dense(tf.reduce_prod(latent_space_shape), activation=dense_activation)(inp)
        x = tf.keras.layers.Reshape(latent_space_shape)(x)
        for _ in range(3):
            x = conv_block(x, nfilters, 3, ndims, normalization='batchnorm')

        latent_space_embedder = tf.keras.Model(inp, x, name='latent_embedder')

        return latent_space_embedder

    def call(self, inp, training=None, autoencode=True):
        if autoencode:
            return super().call(inp, training=training)
        else:
            return self.decoder(self.latent_space_embedder(inp, training=training), training=training)

    @staticmethod
    def unify_metrics(autoencoder_metrics, latent_embedder_metrics):
        metrics_dict = {}
        for metric in autoencoder_metrics:
            metrics_dict[metric+'_AE'] = autoencoder_metrics[metric]
        for metric in latent_embedder_metrics:
            metrics_dict[metric+'_LE'] = latent_embedder_metrics[metric]
        return metrics_dict
        

    def train_step(self, x):

        sensor_inputs, full_fields = x

        self.latent_space_embedder.trainable=False
        autoencoder_metrics = super().train_step(full_fields)
        self.latent_space_embedder.trainable=True

        
        for _ in range(self.latent_space_step_ratio):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.latent_space_embedder.trainable_variables)
                latent_space_pred = self(sensor_inputs, autoencode=False, training=True)
                latent_space_loss = self.compute_loss(sensor_inputs, full_fields, latent_space_pred)
            self.l_optimizer.minimize(latent_space_loss, self.latent_space_embedder.trainable_variables, tape=tape)
        latent_embedder_metrics = self.compute_metrics(sensor_inputs, full_fields, latent_space_pred, None)
            
        return self.unify_metrics(autoencoder_metrics, latent_embedder_metrics)

    def test_step(self, x):

        sensor_inputs, full_fields = x

        autoencoder_metrics = super().test_step(full_fields)
        
        latent_space_pred = self(sensor_inputs, autoencode=False, training=False)
        latent_space_loss = self.compute_loss(sensor_inputs, full_fields, latent_space_pred)
        latent_embedder_metrics = self.compute_metrics(sensor_inputs, full_fields, latent_space_pred, None)

        return self.unify_metrics(autoencoder_metrics, latent_embedder_metrics)

    def compile(self, l_optimizer, latent_space_step_ratio = 1, *args, **kwargs):
        self.l_optimizer = l_optimizer
        self.latent_space_step_ratio = latent_space_step_ratio
        super().compile(*args, **kwargs)
        
        

if __name__ == '__main__':

    mod = ConvAutoencoder(levels=4,
                          base_filters=32,
                          normalization='batchnorm',
                          activation=tf.nn.leaky_relu,
                          pre_activation=True,
                          residual_connection=True,
                          input_shape=[64,64,64,3]
                          )

    mod.save_weights('/tmp/test.h5')

    
    import pdb; pdb.set_trace()

    
    
    
