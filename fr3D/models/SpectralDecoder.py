import copy
import tensorflow as tf

from .FNO import SpectralConv, SpectralConv_RealFullFFT, FNO

class SpectralDecoderLayer(SpectralConv):
    def __init__(self,output_shape,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
        self.n_fft_corners = 2**(self.ndims-1)
        dense_units = tf.reduce_prod(self.modes)*self.n_fft_corners*(self.out_channels)
        
        self.dense_real = tf.keras.layers.Dense(dense_units, activation=self.activation)
        self.dense_imag = tf.keras.layers.Dense(dense_units, activation=self.activation)
        
        assert len(output_shape)==self.ndims
        self.oshape = tuple(output_shape)
        self.rfft_out_size = SpectralConv_RealFullFFT._get_rfft_out_size(self.oshape, n_leading_dims=0)
        starts = tf.zeros((self.ndims,), dtype=tf.int32) + self._wr_slicing_modes * (self.oshape - self._modes_tensor)
        starts = tf.concat([tf.zeros((2**(self.ndims-1), 2), dtype=tf.int32), starts],1)
        scatter_starts = starts[:,2:]
        scatter_ends = (starts + self._wr_slice_sizes)[:,2:]
        self.scatter_indices = self._build_scatter_indices(scatter_starts, scatter_ends)
        
    def build(self, input_shape):
        self.built=True
        
    def compute_output_shape(self, input_shape):
        if tf.keras.backend.image_data_format() == 'channels_last':
            return tf.TensorShape([self.input_shape[0], *self.oshape, self.out_channels])
        else: 
            return tf.TensorShape([self.input_shape[0], self.out_channels, *self.oshape])
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_shape': self.oshape,
        })
        return config
        
    def _rfft_result(self, x):
        
        xshape = tf.shape(x)
        bsize = xshape[0]
        
        fft_tensor_size = tf.concat([self.rfft_out_size, [bsize, self.out_channels]],0)
        
        Re = self.dense_real(x)
        Im = self.dense_imag(x)
        
        y = tf.dtypes.complex(Re, Im)
        y = tf.reshape(y, [bsize, self.n_fft_corners, self.out_channels, *self.modes])
        y = tf.transpose(y, [1,*list(range(3,3+len(self.modes))),0,2])
        y = tf.scatter_nd(self.scatter_indices, y, fft_tensor_size)
        y = tf.transpose(y, [self.ndims,self.ndims+1, *list(range(self.ndims))])
        return y
    
    def call(self, x):
        out_ft = self._rfft_result(x)
        out = self._fft_funcs[self.ndims]['irfft'](out_ft, fft_length=self.oshape)
        
        if tf.keras.backend.image_data_format() == 'channels_last':
            out = tf.transpose(out, self._image_data_format_rev_transpose_idxs)
            
        return self.activation(out)
 
class SpectralDecoderLayer_RealFullFFT(SpectralDecoderLayer):
    _fft_funcs = SpectralConv_RealFullFFT._fft_funcs
    _rfft_to_fft_funcs = SpectralConv_RealFullFFT._rfft_to_fft_funcs
    
    def call(self, x):
        xshape = tf.shape(x)
        final_dim_even_length = tf.logical_not(tf.cast(xshape[-1 if tf.keras.backend.image_data_format()=='channels_first' else -2]%2, tf.bool))
        
        out_rfft = self._rfft_result(x)
        
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

def SpectralDecoder(grid_shape, inp, out_channels, hidden_layer_channels, modes, hidden_layer_activations=None, return_layer=False, **fno_config):
    
    if isinstance(inp, int) or isinstance(inp, tuple) or isinstance(inp, list):
        inp = tf.keras.layers.Input(inp)
    
    if len(grid_shape) == 3:
        x = SpectralDecoderLayer_RealFullFFT(grid_shape, modes=modes, out_channels=hidden_layer_channels, activation=hidden_layer_activations)(inp)
    elif len(grid_shape) in [1,2]:
        x = SpectralDecoderLayer(grid_shape, modes=modes, out_channels=hidden_layer_channels, activation=hidden_layer_activations)(inp)
    else:
        raise(ValueError(f'grid_shape must contain 1,2 or 3 dimensions; got {len(grid_shape)}'))

    x = FNO(x, out_channels, hidden_layer_channels, modes, hidden_layer_activations=hidden_layer_activations, return_layer=True, **fno_config)

    if return_layer:
        return x
    else:
        model=tf.keras.Model(inp,x,name="SpectralDecoder")
        return model
    
        
if __name__ == '__main__':
    m = SpectralDecoder([64,64,64], 100, 2, 16, [8,8,8], hidden_layer_activations=tf.nn.leaky_relu, final_activation=None)
    m.summary()

    x = tf.random.uniform((2,100))
    t = tf.random.uniform((2,64,64,64,2))

    with tf.GradientTape() as tape:
        y = m(x)
        l = tf.reduce_mean(tf.abs(y-t))
    g = tape.gradient(l, m.trainable_variables)
    
        
        
        
        
        
