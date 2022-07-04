import numpy as np
import tensorflow as tf

from .ShallowDecoder import _get_original_shallow_decoder_layers
from .UNet import UNet

def convert_to_keras_tensors(layers_list):
    keras_tensors = [layers_list[0]]
    for layer in layers_list[1:]:
        keras_tensors.append(layer(keras_tensors[-1]))
    return keras_tensors

class SD_UNet(tf.keras.models.Model):
    def __init__(self, input_units, grid_shape, out_channels = 1, sd_config = None, unet_config = None):
        
        if sd_config is None:
            sd_config = {}
        sd_final_units = int(np.prod(grid_shape))
        shallow_decoder_layers = _get_original_shallow_decoder_layers(input_units, sd_final_units, **sd_config)
        shallow_decoder_keras_tensors = convert_to_keras_tensors(shallow_decoder_layers)
        
        unet_input_shape = [1,*grid_shape] if tf.keras.backend.image_data_format() == 'channels_first' else [*grid_shape,1]
        unet_input_reshape_layer = tf.keras.layers.Reshape(unet_input_shape)(shallow_decoder_keras_tensors[-1])

        if unet_config is None:
            unet_config = {}
        unet_final_layer = UNet(input_layer = unet_input_reshape_layer, nx = None, ny = None, in_channels=None, out_channels=out_channels, padding='same', return_layer = True, **unet_config)

        super().__init__(shallow_decoder_keras_tensors[0], unet_final_layer)
