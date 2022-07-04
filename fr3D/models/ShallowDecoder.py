import copy
import tensorflow as tf

def _get_original_shallow_decoder_layers(input_layer_shape, output_layer_size, hidden_layer_units = None, hidden_layer_activations = None, normalization = None, l2_regularization = 0.0):
    if hidden_layer_units is None:
        hidden_layer_units = [40,40]
    if hidden_layer_activations is None:
        hidden_layer_activations = ['relu' for _ in hidden_layer_units]
    else:
        for k in range(len(hidden_layer_activations)):
            activation = hidden_layer_activations[k]
            if activation == 'leaky_relu':
                activation = tf.nn.leaky_relu
            hidden_layer_activations[k] = activation
        
    assert len(hidden_layer_activations) == len(hidden_layer_units)

    normalization_map = {'batchnorm': tf.keras.layers.BatchNormalization, 'layernorm': tf.keras.layers.LayerNormalization}

    layers = [tf.keras.layers.Input(input_layer_shape)]
    for units, activation in zip(hidden_layer_units, hidden_layer_activations):
        regularizer = tf.keras.regularizers.L2(l2_regularization) if abs(l2_regularization) > 0.0 else None
        layers.append(tf.keras.layers.Dense(units, activation = activation, kernel_initializer='glorot_normal', kernel_regularizer = regularizer, bias_regularizer = regularizer))
        if normalization is not None:
            layers.append(normalization_map[normalization]())
    layers.append(tf.keras.layers.Dense(output_layer_size))
    return layers

def original_shallow_decoder(input_layer_shape, output_layer_size, learning_rate=None, hidden_layer_units = None, hidden_layer_activations = None, normalization = None, l2_regularization = 0.0, loss_function = None, metrics = None):

    layers = _get_original_shallow_decoder_layers(input_layer_shape, output_layer_size, hidden_layer_units, hidden_layer_activations, normalization, l2_regularization)

    model = tf.keras.models.Sequential(layers)

    
    # Optimiser
    if learning_rate is not None:
        optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Defining losses
    if loss_function is None:
        loss_function = tf.keras.losses.MeanSquaredError()

    # Defining metrics
    try:
        model.compile(optimizer=optimiser, loss=loss_function, metrics=metrics)
    except:
        pass

    return model

if __name__ == '__main__':
    m = original_shallow_decoder(18, 20000, 1e-4, normalization='layernorm')
    m.summary()
