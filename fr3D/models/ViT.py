#based on https://keras.io/examples/vision/masked_image_modeling/
import tensorflow as tf
import numpy as np

def mlp(x, dropout_rate, hidden_units):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x

def handle_input_spec(input_spec):
    if isinstance(input_spec, int):
        projection_dim = input_spec
        inputs = tf.keras.layers.Input((None, projection_dim))
    elif isinstance(input_spec, (list, tuple)):
        assert len(input_spec) == 2
        input_spec = tuple(input_spec)
        projection_dim = input_spec[-1]
        inputs = tf.keras.layers.Input(input_spec)
    else:
        inputs = input_spec
        projection_dim = inputs.shape[-1]

    return inputs, projection_dim

def get_patch_dimensionality(n_channels, n_dims, patch_size):
    return np.prod([n_channels] + [patch_size]*n_dims)

def ViTEncoder(num_heads, num_layers, input_spec):

    inputs, projection_dim = handle_input_spec(input_spec)
    
    LAYER_NORM_EPS=1e-6
    ENC_TRANSFORMER_UNITS = [projection_dim*2, projection_dim]
    
    x = inputs

    for _ in range(num_layers):
        # Layer normalization 1.
        x1 = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x)

        # Create a multi-head attention layer.
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        x2 = tf.keras.layers.Add()([attention_output, x])

        # Layer normalization 2.
        x3 = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=ENC_TRANSFORMER_UNITS, dropout_rate=0.1)

        # Skip connection 2.
        x = tf.keras.layers.Add()([x3, x2])

    outputs = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x)
    return tf.keras.Model(inputs, outputs, name="ViTEncoder")

def ViTDecoder(input_spec, patch_size, num_heads, num_layers, n_channels=None, n_dims=None, final_activation=None, output_reshape=None):

    inputs, projection_dim = handle_input_spec(input_spec)

    if output_reshape is not None:
        n_channels = output_reshape[1 if tf.keras.backend.image_data_format()=='channels_first' else -1]
        n_dims = len(output_reshape)-1
    patch_dimensionality = get_patch_dimensionality(n_channels, n_dims, patch_size)
    
    LAYER_NORM_EPS=1e-6
    DEC_TRANSFORMER_UNITS = [projection_dim*2, projection_dim]
    DEC_PROJECTION_DIM = projection_dim
    
    x = tf.keras.layers.Dense(DEC_PROJECTION_DIM)(inputs)

    for _ in range(num_layers):
        # Layer normalization 1.
        x1 = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x)

        # Create a multi-head attention layer.
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=DEC_PROJECTION_DIM, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        x2 = tf.keras.layers.Add()([attention_output, x])

        # Layer normalization 2.
        x3 = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=DEC_TRANSFORMER_UNITS, dropout_rate=0.1)

        # Skip connection 2.
        x = tf.keras.layers.Add()([x3, x2])

    
    x = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x)
    pre_final = tf.keras.layers.Dense(units=patch_dimensionality, activation=final_activation)(x)

    if output_reshape is not None:
        outputs = tf.keras.layers.Reshape(output_shape)(pre_final)
    else:
        outputs = pre_final

    return tf.keras.Model(inputs, outputs, name="ViTDecoder")


from ..layers import Patches, MaskedPatchEncoder
class ViTFR(tf.keras.models.Model):
    def __init__(self, output_shape: tuple[int], patch_size: int, embedding_size: int, encoder_heads: int, encoder_layers: int, decoder_heads: int, decoder_layers: int, mask_proportion=0.75, final_activation: str = None):

        super().__init__()
        
        self.patchifier = tf.keras.Sequential([tf.keras.layers.InputLayer(output_shape),
                                               Patches(patch_size),
                                               MaskedPatchEncoder(embedding_size, mask_proportion)],
                                              name='patchifier')

        self.embedding_size = embedding_size
        #self.sensor_embedder = tf.keras.layers.Dense(embedding_size)

        encoder_input_shape = list(self.patchifier.output_shape[1:])
        encoder_input_shape[0] += 1
        self.encoder = ViTEncoder(encoder_heads,
                                  encoder_layers,
                                  encoder_input_shape)

        input_channels = output_shape[1 if tf.keras.backend.image_data_format()=='channels_first' else -1]
        input_dims = len(output_shape)-1
        self.decoder = ViTDecoder(self.encoder.output_shape[1:],
                                  patch_size,
                                  decoder_heads,
                                  decoder_layers,
                                  n_channels = input_channels,
                                  n_dims = input_dims,
                                  final_activation=final_activation)

        decoder_output_shape = list(self.decoder.output_shape)
        decoder_output_shape[1] -= 1
        self.final_reshape = tf.keras.Sequential([tf.keras.layers.InputLayer(decoder_output_shape[1:]),
                                                  tf.keras.layers.Reshape(output_shape)],
                                                 name='final_reshape')

    def build(self, x):
        sensor_input_shape, full_field_shape = x

        self.sensor_embedder = tf.keras.Sequential([tf.keras.layers.InputLayer(sensor_input_shape[1:]),
                                                    tf.keras.layers.Dense(self.embedding_size)],
                                                   name='sensor_embedder')

    def call(self, x):
        sensor_inputs, full_fields = x

        sensor_embedding = self.sensor_embedder(sensor_inputs)
        patch_embeddings = self.patchifier(full_fields)

        encoder_input = tf.concat([sensor_embedding[:, tf.newaxis, :], patch_embeddings], 1)

        encoder_output = self.encoder(encoder_input)

        decoder_output = self.decoder(encoder_output)

        return self.final_reshape(decoder_output[:,1:,:])
        
    def train_step(self, x):

        sensor_inputs, full_fields = x

        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            pred = self(x)
            loss_val = self.compiled_loss(full_fields, pred)
        grads = tape.gradient(loss_val, self.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Report progress.
        self.compiled_metrics.update_state(full_fields, pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, x):

        sensor_inputs, full_fields = x
        pred = self(x)
        loss_val = self.compiled_loss(full_fields, pred)

        # Report progress.
        self.compiled_metrics.update_state(full_fields, pred)
        return {m.name: m.result() for m in self.metrics}
        
        
    

if __name__ == '__main__':

    oshape = (2,64,64,64,2)
    sshape = (2,875)

    o = tf.random.uniform(oshape)
    s = tf.random.uniform(sshape)
    
    mod = ViTFR(oshape[1:], 8, 768, 4, 6, 4, 2)
    zz = mod([s,o])
    mod.summary()
