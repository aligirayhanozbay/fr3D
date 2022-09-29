import numpy as np
import tensorflow as tf
try:
    from tensorflow.python.keras.engine.keras_tensor import KerasTensor as KerasTensor_tf
    from keras.engine.keras_tensor import KerasTensor
except:
    from tensorflow.python.keras.engine.keras_tensor import KerasTensor as KerasTensor
import itertools
import copy
import warnings

# Li Z, Kovachki N, Azizzadenesheli K, Liu B, Bhattacharya K, Stuart A, Anandkumar A. Fourier neural operator for parametric partial differential equations. arXiv preprint arXiv:2010.08895. 2020 Oct 18.


def complex_uniform_initializer(scale=0.05):
    real_initializer = tf.keras.initializers.RandomUniform(-scale,scale)
    def initializer(shape,dtype):
        if dtype == tf.complex64:
            dtype = tf.float32
        elif dtype == tf.complex128:
            dtype = tf.float64
        real = real_initializer(shape,dtype)
        imag = real_initializer(shape,dtype)
        return tf.dtypes.complex(real,imag)
    return initializer

@tf.function(experimental_relax_shapes=True)
def _rfft_tensor_to_fft_1d(w, n_leading_dims = 0, final_dim_even_length=False):
    rank_w = tf.rank(w)
    shape_w = tf.shape(w)
    even_length_wsize_reduction = tf.cast(final_dim_even_length, shape_w.dtype)
    wsize = shape_w[-1] - 1 - even_length_wsize_reduction
    wstar = tf.reverse(tf.math.conj(w[...,1:1+wsize]),[-1])
    wp = tf.concat([w,wstar],-1)
    return wp

@tf.function(experimental_relax_shapes=True)
def _rfft_tensor_to_fft_2d(w, n_leading_dims = 0, final_dim_even_length=False):
    #Illustration of internal variables for a 6x4 FFT
    #* * * * * * -> w0p
    #-----------
    #*|* *|*|* *
    #*|* *|*|* * -> w1p
    #*|* *|*|* *
    #|  w1 |
    #|     ----> middle_terms
    #----------> leading_terms
    rank_w = tf.rank(w)
    shape_w = tf.shape(w)
    even_length_wsize_reduction = tf.cast(final_dim_even_length, shape_w.dtype)
    wsize = shape_w[-1] - 1 - even_length_wsize_reduction

    w0_sl_start = tf.zeros((rank_w,), dtype=shape_w.dtype)
    w0_sl_size = tf.concat([
        shape_w[:n_leading_dims],
        [1],
        shape_w[n_leading_dims+1:]
    ],0)
    w0 = tf.slice(w, w0_sl_start, w0_sl_size)
    w0p = _rfft_tensor_to_fft_1d(tf.squeeze(w0,n_leading_dims), n_leading_dims = n_leading_dims, final_dim_even_length = final_dim_even_length)
    w0p = tf.expand_dims(w0p, n_leading_dims)

    w1_sl_start = tf.concat([
        tf.zeros((n_leading_dims,), dtype = shape_w.dtype),
        [1],
        tf.zeros((rank_w-n_leading_dims-2), dtype=shape_w.dtype),
        [1]
    ],0)
    w1_sl_size = tf.concat([
        shape_w[:n_leading_dims],
        [shape_w[n_leading_dims]-1],
        shape_w[n_leading_dims+1:-1],
        [wsize]
    ],0)
    w1 = tf.slice(w, w1_sl_start, w1_sl_size)
    rev_order = tf.range(n_leading_dims, rank_w)
    w1p = tf.math.conj(tf.reverse(w1, rev_order))

    leading_terms_sl_start = tf.concat([
        tf.zeros((n_leading_dims,), dtype = shape_w.dtype),
        [1],
        tf.zeros((rank_w-n_leading_dims-2), dtype=shape_w.dtype),
        [0]
    ],0)
    leading_terms_sl_size = tf.concat([
        shape_w[:n_leading_dims],
        [shape_w[n_leading_dims]-1],
        shape_w[n_leading_dims+1:-1],
        [1]
    ],0)
    leading_terms = tf.slice(w, leading_terms_sl_start, leading_terms_sl_size)

    middle_terms_sl_start = tf.concat([
        tf.zeros((n_leading_dims,), dtype = shape_w.dtype),
        [1],
        tf.zeros((rank_w-n_leading_dims-2), dtype=shape_w.dtype),
        [1+wsize]
    ],0)
    middle_terms_sl_size = tf.concat([
        shape_w[:n_leading_dims],
        [shape_w[n_leading_dims]-1],
        shape_w[n_leading_dims+1:-1],
        [even_length_wsize_reduction]
    ],0)
    middle_terms = tf.slice(w, middle_terms_sl_start, middle_terms_sl_size)

    return tf.concat([w0p,tf.concat([leading_terms, w1, middle_terms, w1p],-1)],n_leading_dims)

@tf.function(experimental_relax_shapes=True)
def _rfft_tensor_to_fft_3d(w, n_leading_dims = 0, final_dim_even_length=False): 
    rank_w = tf.rank(w)
    shape_w = tf.shape(w)
    even_length_wsize_reduction = tf.cast(final_dim_even_length, shape_w.dtype)
    wsize = shape_w[-1] - 1 - even_length_wsize_reduction

    w0_sl_start = tf.zeros((rank_w,), dtype=shape_w.dtype)
    w0_sl_size = tf.concat([
        shape_w[:n_leading_dims],
        [1],
        shape_w[n_leading_dims+1:]
    ],0)
    w0 = tf.slice(w, w0_sl_start, w0_sl_size)
    w0p = _rfft_tensor_to_fft_2d(tf.squeeze(w0,n_leading_dims), n_leading_dims = n_leading_dims, final_dim_even_length = final_dim_even_length)
    w0p = tf.expand_dims(w0p, n_leading_dims)

    w1_sl_start = tf.concat([
        tf.zeros((n_leading_dims,), dtype = shape_w.dtype),
        [1],
        tf.zeros((rank_w-n_leading_dims-2), dtype=shape_w.dtype),
        [1]
    ],0)
    w1_sl_size = tf.concat([
        shape_w[:n_leading_dims],
        [shape_w[n_leading_dims]-1],
        shape_w[n_leading_dims+1:-1],
        [wsize]
    ],0)
    w1 = tf.slice(w, w1_sl_start, w1_sl_size)
    rev_order = [4,2]
    w1p = tf.math.conj(tf.reverse(w1, rev_order))
    w1p = tf.concat([w1p[...,:,:1,:],tf.reverse(w1p[...,:,1:,:],[3])],3)

    leading_terms_sl_start = tf.concat([
        tf.zeros((n_leading_dims,), dtype = shape_w.dtype),
        [1],
        tf.zeros((rank_w-n_leading_dims-2), dtype=shape_w.dtype),
        [0]
    ],0)
    leading_terms_sl_size = tf.concat([
        shape_w[:n_leading_dims],
        [shape_w[n_leading_dims]-1],
        shape_w[n_leading_dims+1:-1],
        [1]
    ],0)
    leading_terms = tf.slice(w, leading_terms_sl_start, leading_terms_sl_size)

    middle_terms_sl_start = tf.concat([
        tf.zeros((n_leading_dims,), dtype = shape_w.dtype),
        [1],
        tf.zeros((rank_w-n_leading_dims-2), dtype=shape_w.dtype),
        [1+wsize]
    ],0)
    middle_terms_sl_size = tf.concat([
        shape_w[:n_leading_dims],
        [shape_w[n_leading_dims]-1],
        shape_w[n_leading_dims+1:-1],
        [even_length_wsize_reduction]
    ],0)
    middle_terms = tf.slice(w, middle_terms_sl_start, middle_terms_sl_size)

    return tf.concat([w0p,tf.concat([leading_terms, w1, middle_terms, w1p],-1)],n_leading_dims)
    

# @tf.function(experimental_relax_shapes=True)
# def _rfft_tensor_to_fft(w, n_leading_dims = 0, final_dim_even_length=False):
#     #recursive version - for 1,2 or 3d. does not work with tf < 2.7 due to bugs.
#     rank_w = tf.rank(w)
#     shape_w = tf.shape(w)
#     even_length_wsize_reduction = tf.cast(final_dim_even_length, shape_w.dtype)
#     wsize = shape_w[-1] - 1 - even_length_wsize_reduction
#     if rank_w == n_leading_dims + 1:
#         wstar = tf.reverse(tf.math.conj(w[...,1:1+wsize]),[-1])
#         wp = tf.concat([w,wstar],-1)
#         return wp
#     elif rank_w > n_leading_dims + 1:
#         w0_sl_start = tf.zeros((rank_w,), dtype=shape_w.dtype)
#         w0_sl_size = tf.concat([
#             shape_w[:n_leading_dims],
#             [1],
#             shape_w[n_leading_dims+1:]
#         ],0)
#         w0 = tf.slice(w, w0_sl_start, w0_sl_size)
#         w0p = _rfft_tensor_to_fft(tf.squeeze(w0,n_leading_dims), n_leading_dims = n_leading_dims, final_dim_even_length = final_dim_even_length)
#         w0p = tf.expand_dims(w0p, n_leading_dims)
        
#         w1_sl_start = tf.concat([
#             tf.zeros((n_leading_dims,), dtype = shape_w.dtype),
#             [1],
#             tf.zeros((rank_w-n_leading_dims-2), dtype=shape_w.dtype),
#             [1]
#         ],0)
#         w1_sl_size = tf.concat([
#             shape_w[:n_leading_dims],
#             [shape_w[n_leading_dims]-1],
#             shape_w[n_leading_dims+1:-1],
#             [wsize]
#         ],0)
#         w1 = tf.slice(w, w1_sl_start, w1_sl_size)
#         if rank_w == n_leading_dims + 2:
#             rev_order = tf.range(n_leading_dims, rank_w)
#             w1p = tf.math.conj(tf.reverse(w1, rev_order))
#         elif rank_w == n_leading_dims + 3:
#             rev_order = [4,2]
#             w1p = tf.math.conj(tf.reverse(w1, rev_order))
#             w1p = tf.concat([w1p[...,:,:1,:],tf.reverse(w1p[...,:,1:,:],[3])],3)
        
#         leading_terms_sl_start = tf.concat([
#             tf.zeros((n_leading_dims,), dtype = shape_w.dtype),
#             [1],
#             tf.zeros((rank_w-n_leading_dims-2), dtype=shape_w.dtype),
#             [0]
#         ],0)
#         leading_terms_sl_size = tf.concat([
#             shape_w[:n_leading_dims],
#             [shape_w[n_leading_dims]-1],
#             shape_w[n_leading_dims+1:-1],
#             [1]
#         ],0)
#         leading_terms = tf.slice(w, leading_terms_sl_start, leading_terms_sl_size)

#         middle_terms_sl_start = tf.concat([
#             tf.zeros((n_leading_dims,), dtype = shape_w.dtype),
#             [1],
#             tf.zeros((rank_w-n_leading_dims-2), dtype=shape_w.dtype),
#             [1+wsize]
#         ],0)
#         middle_terms_sl_size = tf.concat([
#             shape_w[:n_leading_dims],
#             [shape_w[n_leading_dims]-1],
#             shape_w[n_leading_dims+1:-1],
#             [even_length_wsize_reduction]
#         ],0)
#         middle_terms = tf.slice(w, middle_terms_sl_start, middle_terms_sl_size)

#         return tf.concat([w0p,tf.concat([leading_terms, w1, middle_terms, w1p],-1)],n_leading_dims)
        
#     else:
#         raise(ValueError('Expected w to have rank > n_leading_dims'))

################################################################
# fourier layer
################################################################
class SpectralConv(tf.keras.layers.Layer):
    _fft_funcs = {
        1:{'rfft': tf.signal.rfft, 'irfft': tf.signal.irfft},
        2:{'rfft': tf.signal.rfft2d, 'irfft': tf.signal.irfft2d},
        3:{'rfft': tf.signal.rfft3d, 'irfft': tf.signal.irfft3d}
    }
    _transpose_indices = {
        1:(2,0,1),
        2:(2,3,0,1),
        3:(3,4,0,1,2)
    }
    def __init__(self, out_channels, modes, activation=None):
        super().__init__()

        self.in_channels = None
        self.out_channels = out_channels
        self.modes = [int(x) for x in modes] #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.ndims = len(self.modes)
        assert (self.ndims > 0) and (self.ndims < 4)

        self._modes_tensor = tf.constant(self.modes, dtype=tf.int32)
        self._wr_slice_sizes = tf.concat([[-1,-1], self._modes_tensor],0)
        self._build_wr_slicing_modes()
        self._image_data_format_transpose_idxs = [0] + [self.ndims+2-1] + list(range(1,1+self.ndims))
        self._image_data_format_rev_transpose_idxs = [0] + list(range(2,2+self.ndims)) + [1]
        
        self.activation = tf.keras.activations.get(activation)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'out_channels': self.out_channels,
            'modes': self.modes,
            'activation': self.activation
        })
        return config

    def _get_weight_shape(self):
        return (2**(self.ndims-1), self.in_channels, self.out_channels, *self.modes)

    def build(self, input_shape):
        if tf.keras.backend.image_data_format() == 'channels_first':
            self.in_channels = input_shape[1]
        else:
            self.in_channels = input_shape[-1]
        self.scale = (1 / (self.in_channels * self.out_channels))
        initializer = complex_uniform_initializer(self.scale)

        var_shape = self._get_weight_shape()

        #cannot use complex weight directly due to tf bug - cant use both complex and float weights
        #in a model when using tf.distribute.MirroredStrategy...
        self.w = self.add_weight(shape = var_shape, initializer = initializer, dtype = tf.complex64, name='w')
        # self.wreal = self.add_weight(shape = var_shape, initializer = initializer, dtype = tf.keras.backend.floatx(), name='wreal')
        # self.wimag = self.add_weight(shape = var_shape, initializer = initializer, dtype = tf.keras.backend.floatx(), name='wimag')
        # self._complex_dtype = 'complex128' if tf.keras.backend.floatx() == 'float64' else 'float32'

    # @property
    # @tf.function
    # def w(self):
    #     return tf.cast(self.wreal, self._complex_dtype) + 1j*tf.cast(self.wimag, self._complex_dtype)

    def compute_output_shape(self, input_shape):
        output_shape = list(copy.deepcopy(input_shape))
        output_shape[1 if tf.keras.backend.image_data_format() == 'channels_first' else -1] = self.out_channels
        return tf.TensorShape(output_shape)

    # Complex multiplication
    @tf.function
    def compl_mul(self, inp, weights):
        # (fft corner, batch, in_channel, x0, x1, ...), (fft corner, in_channel, out_channel, x0, x1, ...) -> (fft corner, batch, out_channel, x0, x1, ...)
        return tf.einsum("wbi...,wio...->w...bo", inp, weights)

    def _build_wr_slicing_modes(self):
        #fft bins are placed at the half of the vertices of an n-dimensional cube.
        n=tf.shape(self._modes_tensor)[0]-1
        i = [0,1]
        modes = np.array(list(itertools.product(*list(itertools.repeat(i,n)))))[:,::-1]
        modes = np.concatenate([modes, np.zeros((modes.shape[0],1), dtype=modes.dtype)],1)
        self._wr_slicing_modes = tf.constant(modes, dtype=tf.int32)


    @tf.function
    def _build_single_scatter_meshgrid(self,starts,ends):
        return tf.stack(tf.meshgrid(*[tf.range(starts[k], ends[k]) for k in range(self.ndims)], indexing='ij'),-1)

    @tf.function
    def _build_scatter_indices(self, starts, ends):
        si = tf.map_fn(lambda x: self._build_single_scatter_meshgrid(x[0], x[1]), (starts, ends), fn_output_signature=tf.int32)
        return si

    @tf.function
    def _wr_slice(self, tensor):
        spatial_dim_sizes = tf.shape(tensor)[2:]
        starts = tf.zeros((self.ndims,), dtype=tf.int32) + self._wr_slicing_modes * (spatial_dim_sizes - self._modes_tensor)
        starts = tf.concat([tf.zeros((2**(self.ndims-1), 2), dtype=tf.int32), starts],1)
        slices = tf.vectorized_map(lambda s: tf.slice(tensor, s, self._wr_slice_sizes), starts)

        scatter_starts = starts[:,2:]
        scatter_ends = (starts + self._wr_slice_sizes)[:,2:]
        scatter_indices = self._build_scatter_indices(scatter_starts, scatter_ends)
        return slices, scatter_indices, spatial_dim_sizes
        
    @tf.function
    def call(self, x):
        if tf.keras.backend.image_data_format() == 'channels_last':
            x = tf.transpose(x, self._image_data_format_transpose_idxs)
        
        xshape = tf.shape(x)
        batchsize = xshape[0]
        
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = self._fft_funcs[self.ndims]['rfft'](x)

        # Multiply relevant Fourier modes
        wr,scatter_ind, fft_sizes = self._wr_slice(x_ft)
        wr = self.compl_mul(wr, self.w)
        fft_tensor_size = tf.concat([fft_sizes, [batchsize, self.out_channels]],0)
        out_ft = tf.scatter_nd(scatter_ind, wr, fft_tensor_size)
        out_ft = tf.transpose(out_ft, self._transpose_indices[self.ndims])

        #Return to physical space
        out = self._fft_funcs[self.ndims]['irfft'](out_ft, fft_length = xshape[2:])

        if tf.keras.backend.image_data_format() == 'channels_last':
            out = tf.transpose(out, self._image_data_format_rev_transpose_idxs)
            
        return self.activation(out)

class SpectralConv_RealFullFFT(SpectralConv):
    #Slower version of the spectral conv using full FFTs.
    #Necessary for 3D since TF does not have gradients for rfft3d/irfft3d...
    _fft_funcs = {
        1:{'fft': tf.signal.fft, 'ifft': tf.signal.ifft},
        2:{'fft': tf.signal.fft2d, 'ifft': tf.signal.ifft2d},
        3:{'fft': tf.signal.fft3d, 'ifft': tf.signal.ifft3d}
    }
    _rfft_to_fft_funcs = {
        1: _rfft_tensor_to_fft_1d,
        2: _rfft_tensor_to_fft_2d,
        3: _rfft_tensor_to_fft_3d
    }

    @staticmethod
    @tf.function
    def _get_rfft_out_size(input_shape, n_leading_dims = 2):
        final_spatial_dim_size = input_shape[-1]
        other_dims_size = input_shape[n_leading_dims:-1]
        return tf.concat([other_dims_size, [(final_spatial_dim_size//2)+1]],0)
            

    def call(self, x):
        x = tf.cast(x, 'complex64')
        
        if tf.keras.backend.image_data_format() == 'channels_last':
            x = tf.transpose(x, self._image_data_format_transpose_idxs)
            
        xshape = tf.shape(x)
        batchsize = xshape[0]
        fft_out_size = xshape[2:]
        final_dim_even_length = tf.logical_not(tf.cast(xshape[-1]%2, tf.bool))
        rfft_out_size = self._get_rfft_out_size(xshape)

        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = self._fft_funcs[self.ndims]['fft'](x)

        wr,scatter_ind,_ = self._wr_slice(x_ft)
        wr = self.compl_mul(wr, self.w)
        rfft_tensor_size = tf.concat([rfft_out_size, [batchsize, self.out_channels]],0)
        out_rfft = tf.scatter_nd(scatter_ind, wr, rfft_tensor_size)
        out_rfft = tf.transpose(out_rfft, self._transpose_indices[self.ndims])

        #Return to physical space
        out_fft = self._rfft_to_fft_funcs[self.ndims](
            out_rfft,
            n_leading_dims = 2,
            final_dim_even_length = final_dim_even_length)
        out = self._fft_funcs[self.ndims]['ifft'](out_fft)
        
        if tf.keras.backend.image_data_format() == 'channels_last':
            out = tf.transpose(out, self._image_data_format_rev_transpose_idxs)

        out = tf.cast(out, tf.keras.backend.floatx())

        return self.activation(out)


class FNOBlock(tf.keras.models.Model):
    _conv_layers = {1: tf.keras.layers.Conv1D,
                    2: tf.keras.layers.Conv2D,
                    3: tf.keras.layers.Conv3D}
    normalization_map = {'batchnorm': tf.keras.layers.BatchNormalization, 'layernorm': tf.keras.layers.LayerNormalization, None: lambda *args,**kwargs: lambda x: x}
    def __init__(self, out_channels, modes, activation=None, conv_kernel_size=1, conv_layer_arguments=None, use_full_ffts=False, normalization = None):
        super().__init__()
        self.normalization = self.normalization_map[normalization](axis=1 if tf.keras.backend.image_data_format() == 'channels_first' else -1)
        if conv_layer_arguments is None:
            conv_layer_arguments = {}
        if use_full_ffts:
            self.spectralconv = SpectralConv_RealFullFFT(out_channels, modes)
        else:
            self.spectralconv = SpectralConv(out_channels, modes)
        self.conv = self._conv_layers[self.spectralconv.ndims](out_channels, conv_kernel_size, padding='same', **conv_layer_arguments)

        if activation is None:
            activation = 'gelu'
        self.activation = tf.keras.activations.get(activation)

    def call(self, x):
        x = self.normalization(x)
        return self.activation(self.spectralconv(x) + self.conv(x))


class PositionalEmbedding(tf.keras.layers.Layer):
    _embedding_types = {'linear': lambda linspace: linspace,
                        'cosine': lambda linspace: tf.cos(np.pi*linspace)}
    def __init__(self, ndims, embedding_type=None):
        super().__init__()
        if embedding_type is None:
            embedding_type = 'linear'
        assert embedding_type in self._embedding_types
        self.embedding_type = self._embedding_types[embedding_type]
        self.ndims = ndims

    def get_config(self):
        config = super().get_config().copy()
        config.update({'ndims': self.ndims, 'embedding_type':self.embedding_type})
        return config

    def call(self, x):
        xshape = tf.shape(x)
        if tf.keras.backend.image_data_format() == 'channels_first':
            spatial_dim_sizes = xshape[2:]
        else:
            spatial_dim_sizes = xshape[1:-1]
        bsize = xshape[0]

        directional_values = [self.embedding_type(tf.linspace(0.0, 1.0, spatial_dim_sizes[k])) for k in range(self.ndims)]
        mg = tf.expand_dims(tf.stack(tf.meshgrid(*directional_values, indexing='ij'),-1 if tf.keras.backend.image_data_format() == 'channels_last' else 0),0)
        mg = tf.tile(mg, [bsize] + [1 for _ in range(self.ndims+1)])

        return tf.concat([x,mg],-1 if tf.keras.backend.image_data_format() == 'channels_last' else 1)

def FNO(inp, out_channels, hidden_layer_channels, modes, hidden_layer_activations = None, conv_kernel_size = 1, conv_layer_arguments=None, n_blocks = 4, dense_activations = None, dense_units = None, positional_embedding_type = None, final_activation = None, return_layer = False):

    ndims = len(modes)

    if dense_activations is None:
        dense_activations = [[None], [None, None]]
    if dense_units is None:
        dense_units = [[],[128]]
    if final_activation is None:
        final_activation = 'linear'
    dense_units[0].append(hidden_layer_channels)
    dense_units[1].append(out_channels)

    if not (isinstance(inp, KerasTensor) or isinstance(inp, KerasTensor_tf)):
        inp = tf.keras.layers.Input(inp)

    x = PositionalEmbedding(ndims, embedding_type = positional_embedding_type)(inp)

    if tf.keras.backend.image_data_format() == 'channels_first':
        forward_transpose_indices = [0] + list(range(2,2+ndims)) + [1]
        backward_transpose_indices = [0,ndims+1] + list(range(1,1+ndims))
        x = tf.transpose(x, forward_transpose_indices)

    for activ,units in zip(dense_activations[0], dense_units[0]):
        x = tf.keras.layers.Dense(units, activation = activ)(x)

    if tf.keras.backend.image_data_format() == 'channels_first':
        x = tf.transpose(x, backward_transpose_indices)

    for _ in range(n_blocks-1):
        x = FNOBlock(hidden_layer_channels, modes, activation = hidden_layer_activations, conv_kernel_size = conv_kernel_size, conv_layer_arguments = conv_layer_arguments, use_full_ffts = len(modes) > 2)(x)

    x = FNOBlock(hidden_layer_channels, modes, activation = final_activation, conv_kernel_size = conv_kernel_size, conv_layer_arguments = conv_layer_arguments, use_full_ffts = len(modes) > 2)(x)

    if tf.keras.backend.image_data_format() == 'channels_first':
        x = tf.transpose(x, forward_transpose_indices)

    for activ,units in zip(dense_activations[1], dense_units[1]):
        x = tf.keras.layers.Dense(units, activation=activ)(x)

    if tf.keras.backend.image_data_format() == 'channels_first':
        x = tf.transpose(x, backward_transpose_indices)

    if return_layer:
        return x
    else:
        model = tf.keras.Model(inp,x)
        return model
    


if __name__ == '__main__':
    tf.keras.backend.set_image_data_format('channels_first')
    bsize = 10
    nc = 4
    no = 2
    ns = [64,59,37]
    modes = [16 for _ in ns]
    if tf.keras.backend.image_data_format() == 'channels_first':
        inpshape = [bsize, nc, *ns]
        outshape = [bsize, no, *ns]
    else:
        inpshape = [bsize, *ns, nc]
        outshape = [bsize, *ns, no]
    inp = tf.random.uniform(inpshape)
    tar = tf.random.uniform(outshape)

    lay1 = SpectralConv(no,modes)
    lay1.build(inpshape)
    lay2 = SpectralConv_RealFullFFT(no,modes)
    lay2.build(inpshape)
    _ = lay1(inp)
    _ = lay2(inp)
    lay2.set_weights(lay1.get_weights())
    
    o1 = lay1(inp)
    o2 = lay2(inp)
    diff = o1 - o2
    mask = tf.math.real(diff * tf.math.conj(diff)) < 1e-4
    check_result = 'Passed' if bool(tf.reduce_all(mask)) else 'Failed'
    print('Full FFT vs RFFT version identical answer check: ' + check_result)
    
    inpl = tf.keras.layers.Input(inpshape[1:])
    mod = FNO(inpl, no, 32, modes)
    mod.summary()

    def get_gradients_function(layer, loss_fn):
        @tf.function
        def grads(x, t):
            y = layer(x)
            loss = loss_fn(t, y)
            return y, tf.gradients(loss, layer.trainable_variables)
        return grads

    grads_fn = get_gradients_function(mod, tf.keras.losses.MeanSquaredError())
    pred, grads = grads_fn(inp,tar)
    check_result = 'Passed' if bool(tf.reduce_all([tf.reduce_all(np.isfinite(g.numpy())) for g in grads])) else 'Failed'
    print('FNO model gradients nan check: ' + check_result)
    if check_result == 'Failed':
        print('Launching debugger')
        import pdb; pdb.set_trace()
    
