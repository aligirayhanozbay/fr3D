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

def get_patch_dimensionality(output_shape, patch_size):
    n_channels = output_shape[1 if tf.keras.backend.image_data_format() == 'channels_first' else -1]
    return np.prod([n_channels] + [patch_size]*(len(output_shape)-1))

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

def ViTDecoder(input_spec, output_shape, patch_size, num_heads, num_layers, final_activation=None):

    inputs, projection_dim = handle_input_spec(input_spec)
    patch_dimensionality = get_patch_dimensionality(output_shape, patch_size)
    
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
    outputs = tf.keras.layers.Reshape(output_shape)(pre_final)

    return tf.keras.Model(inputs, outputs, name="ViTDecoder")

if __name__ == '__main__':
    from ..layers import Patches, PatchEncoder
    psize = 8
    x = tf.keras.layers.Input(shape=(64,64,64,3))
    m = Patches(psize)(x)
    m = PatchEncoder(768)(m)
    print(m)
    
    enc = ViTEncoder(4,6,m)
    enc.summary()

    dec = ViTDecoder(enc.output_shape[1:], x.shape[1:], psize, 4, 2)
    dec.summary()
    
