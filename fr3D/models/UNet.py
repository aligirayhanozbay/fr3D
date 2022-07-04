from typing import Optional, Union, Callable, List

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.optimizers import Adam

def _get_filter_count(layer_idx, filters_root):
    return 2 ** layer_idx * filters_root


def _get_kernel_initializer(filters, kernel_size, ndims=3):
    stddev = np.sqrt(2 / (kernel_size ** ndims * filters))
    return TruncatedNormal(stddev=stddev)

def _get_padding_sizes(kernel_size, padding='same', include_time_dim = False, spatial_dims_only=False):
    data_format = tf.keras.backend.image_data_format()

    batch_dim_paddings = [[0,0],[0,0]] if include_time_dim else [[0,0]]
    channel_dim_paddings = [0,0]
    if padding == 'same':
        spatial_dim_paddings = [[k//2, k//2-(1-k%2)] for k in kernel_size]
    elif padding == 'valid':
        spatial_dim_paddings = [[0,0] for _ in kernel_size]
    else:
        raise(ValueError('padding must be same or valid'))

    if spatial_dims_only:
        return spatial_dim_paddings
    else:
        if data_format == 'channels_first':
            paddings = [*batch_dim_paddings, channel_dim_paddings] + spatial_dim_paddings
        elif data_format == 'channels_last':
            paddings = [*batch_dim_paddings] + spatial_dim_paddings + [channel_dim_paddings]
        else:
            raise(ValueError('expecting channels_first or channels_last as tf keras image data format'))

        return paddings

def _get_activation_layer(activ, **kwargs):
    if activ in tf.keras.layers.Layer.__subclasses__():
        activ_layer = activ(**kwargs)
    else:
        activ_layer = layers.Activation(activ)
    return activ_layer

class ConvBlock(layers.Layer):
    normalization_map = {'batchnorm': tf.keras.layers.BatchNormalization, 'layernorm': tf.keras.layers.LayerNormalization, None: lambda *args,**kwargs: lambda x: x}
    def __init__(self, layer_idx, filters_root, kernel_size, dropout_rate, padding, activation, normalization = None, padding_mode = None, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.layer_idx=layer_idx
        self.filters_root=filters_root
        self.kernel_size=kernel_size
        self.dropout_rate=dropout_rate
        self.activation=activation

        assert (padding.lower() in ['same', 'valid'])
        self.padding=padding.lower()
        assert (padding_mode is None) or (padding_mode.lower() in ['constant', 'reflect', 'symmetric'])
        self.padding_mode = padding_mode.lower() if padding_mode is not None else 'constant'

        filters = _get_filter_count(layer_idx, filters_root)
        self.normalization_1 = self.normalization_map[normalization](axis=1 if tf.keras.backend.image_data_format() == 'channels_first' else -1)
        self.conv_1 = layers.Conv3D(filters=filters,
                                      kernel_size=(kernel_size, kernel_size, kernel_size),
                                      kernel_initializer=_get_kernel_initializer(filters, kernel_size),
                                      strides=1,
                                      padding="valid")
        self.dropout_1 = layers.Dropout(rate=dropout_rate)
        self.activation_1 = _get_activation_layer(activation)
        self._conv1_paddings_size = _get_padding_sizes(self.conv_1.kernel_size)

        self.normalization_2 = self.normalization_map[normalization](axis=1 if tf.keras.backend.image_data_format() == 'channels_first' else -1)
        self.conv_2 = layers.Conv3D(filters=filters,
                                      kernel_size=(kernel_size, kernel_size, kernel_size),
                                      kernel_initializer=_get_kernel_initializer(filters, kernel_size),
                                      strides=1,
                                      padding="valid")
        self.dropout_2 = layers.Dropout(rate=dropout_rate)
        self.activation_2 = _get_activation_layer(activation)
        self._conv2_paddings_size = _get_padding_sizes(self.conv_2.kernel_size)
        
    def _pad(self, x, paddings):
        return tf.pad(x, paddings, mode=self.padding_mode)
            
        
    def call(self, inputs, training=None, **kwargs):
        x = inputs

        x = self.normalization_1(x)
        x = self.conv_1(x)
        x = self._pad(x, self._conv1_paddings_size)
        if training:
            x = self.dropout_1(x)
        x = self.activation_1(x)
        
        x = self.normalization_2(x)
        x = self.conv_2(x)
        x = self._pad(x, self._conv2_paddings_size)
        if training:
            x = self.dropout_2(x)
        x = self.activation_2(x)
        
        return x

    def get_config(self):
        return dict(layer_idx=self.layer_idx,
                    filters_root=self.filters_root,
                    kernel_size=self.kernel_size,
                    dropout_rate=self.dropout_rate,
                    padding=self.padding,
                    activation=self.activation,
                    **super(ConvBlock, self).get_config(),
                    )


class UpconvBlock(layers.Layer):
    normalization_map = {'batchnorm': tf.keras.layers.BatchNormalization, 'layernorm': tf.keras.layers.LayerNormalization, None: lambda *args,**kwargs: lambda x: x}
    def __init__(self, layer_idx, filters_root, kernel_size, pool_size, padding, activation, normalization = None, **kwargs):
        super(UpconvBlock, self).__init__(**kwargs)
        self.layer_idx=layer_idx
        self.filters_root=filters_root
        self.kernel_size=kernel_size
        self.pool_size=pool_size
        self.padding=padding
        self.activation=activation

        filters = _get_filter_count(layer_idx + 1, filters_root)
        self.normalization = self.normalization_map[normalization](axis=1 if tf.keras.backend.image_data_format() == 'channels_first' else -1)
        self.upconv = layers.Conv3DTranspose(filters // 2,
                                             kernel_size=(pool_size, pool_size, pool_size),
                                             kernel_initializer=_get_kernel_initializer(filters, kernel_size),
                                             strides=pool_size, padding=padding)

        self.activation_1 = _get_activation_layer(activation)

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.normalization(x)
        x = self.upconv(x)
        x = self.activation_1(x)
        return x

    def get_config(self):
        return dict(layer_idx=self.layer_idx,
                    filters_root=self.filters_root,
                    kernel_size=self.kernel_size,
                    pool_size=self.pool_size,
                    padding=self.padding,
                    activation=self.activation,
                    **super(UpconvBlock, self).get_config(),
                    )

class CropConcatBlock(layers.Layer):

    def call(self, x, down_layer, **kwargs):
        x1_shape = tf.shape(down_layer)
        x2_shape = tf.shape(x)

        if tf.keras.backend.image_data_format() == 'channels_last':
            height_diff = (x1_shape[1] - x2_shape[1]) // 2
            width_diff = (x1_shape[2] - x2_shape[2]) // 2
            depth_diff = (x1_shape[3] - x2_shape[3]) // 2
            down_layer_cropped = down_layer[:,
                                            height_diff: (x1_shape[1] - height_diff),
                                            width_diff: (x1_shape[2] - width_diff),
                                            depth_diff: (x1_shape[3] - depth_diff),
                                            :]
        else:
            height_diff = (x1_shape[2] - x2_shape[2]) // 2
            width_diff = (x1_shape[3] - x2_shape[3]) // 2
            depth_diff = (x1_shape[4] - x2_shape[4]) // 2
            down_layer_cropped = down_layer[:,:,
                                            height_diff: (x1_shape[2] - height_diff),
                                            width_diff: (x1_shape[3] - width_diff),
                                            depth_diff: (x1_shape[4] - depth_diff)
                                            ]

        x = tf.concat([down_layer_cropped, x], axis=-1 if tf.keras.backend.image_data_format() == 'channels_last' else 1)
        return x
    
def UNet(
        input_layer: Optional[tf.keras.layers.Layer] = None,
        nx: Optional[int] = None,
        ny: Optional[int] = None,
        nz: Optional[int] = None,
        in_channels: int = 1,
        out_channels: int = 1,
        layer_depth: int = 5,
        filters_root: int = 64,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout_rate: int = 0.5,
        padding:str="valid",
        padding_mode:str="constant",
        activation:Union[str, Callable]="relu",
        final_activation:Union[str,Callable]="linear",
        final_kernel_size: int = None,
        return_layer=False,
        normalization=None):
    """
    Constructs a U-Net model
    :param nx: (Optional) image size on x-axis
    :param ny: (Optional) image size on y-axis
    :param nz: (Optional) image size on z-axis
    :param in_channels: number of input channels of the input tensors
    :param out_channels: number of channels of the output
    :param layer_depth: total depth of unet
    :param filters_root: number of filters in top unet layer
    :param kernel_size: size of convolutional layers
    :param pool_size: size of maxplool layers
    :param dropout_rate: rate of dropout
    :param padding: padding to be used in convolutions
    :param activation: activation to be used
    :return: A TF Keras model
    """
    if input_layer is None:
        inpshape = (nx, ny, nz, in_channels) if tf.keras.backend.image_data_format() == 'channels_last' else (in_channels, nx, ny, nz)
        inputs = Input(shape=inpshape, name="inputs")
        x = inputs
    else:
        x = input_layer

    if activation == 'leaky_relu':
        activation = tf.nn.leaky_relu
    elif activation == 'prelu':
        activation = tf.keras.layers.PReLU

    if final_kernel_size is None:
        final_kernel_size = kernel_size
        
    contracting_layers = {}

    conv_params = dict(filters_root=filters_root,
                       kernel_size=kernel_size,
                       dropout_rate=dropout_rate,
                       padding=padding,
                       padding_mode=padding_mode,
                       activation=activation,
                       normalization=normalization)

    for layer_idx in range(0, layer_depth - 1):
        x = ConvBlock(layer_idx, **conv_params)(x)
        contracting_layers[layer_idx] = x
        x = layers.AveragePooling3D((pool_size, pool_size, pool_size), padding = 'same', data_format = tf.keras.backend.image_data_format())(x)

    x = ConvBlock(layer_idx + 1, **conv_params)(x)

    for layer_idx in range(layer_idx, -1, -1):
        x = UpconvBlock(layer_idx,
                        filters_root,
                        kernel_size,
                        pool_size,
                        padding,
                        activation,
                        normalization=normalization)(x)
        x = CropConcatBlock()(x, contracting_layers[layer_idx])
        x = ConvBlock(layer_idx, **conv_params)(x)

    x = layers.Conv3D(filters=out_channels,
                      kernel_size=final_kernel_size,
                      kernel_initializer=_get_kernel_initializer(filters_root, final_kernel_size),
                      strides=1,
                      padding=padding)(x)

    
    outputs = layers.Activation(final_activation, name="outputs")(x)

    if return_layer:
        return outputs
    else:
        model = Model(inputs, outputs, name="unet")
        return model


if __name__ == '__main__':
    mod = UNet(nx=64, ny=64, nz=64, in_channels=3, out_channels=1, layer_depth=4, filters_root=64, kernel_size=3, activation='prelu', padding="same")

    inp = tf.random.uniform((10,64,64,64,3))
    out = mod(inp)
    import pdb; pdb.set_trace()
    z=1
