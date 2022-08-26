import copy
import tensorflow as tf

from .FNO import SpectralConv, SpectralConv_RealFullFFT, FNO

class SpectralDecoder(SpectralConv):
    def __init__(self,output_shape,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
        self.n_fft_corners = 2**(self.ndims-1)
        dense_units = tf.reduce_prod(self.modes)*self.n_fft_corners*(self.out_channels)
        
        self.dense_real = tf.keras.layers.Dense(dense_units)
        self.dense_imag = tf.keras.layers.Dense(dense_units)
        
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
 
class SpectralDecoder_RealFullFFT(SpectralDecoder):
    _fft_funcs = SpectralConv_RealFullFFT._fft_funcs
    _rfft_to_fft_funcs = SpectralConv_RealFullFFT._rfft_to_fft_funcs
    
    def call(self, x):
        xshape = tf.shape(x)
        final_dim_even_length = tf.logical_not(tf.cast(xshape[-1]%2, tf.bool))
        
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
        
if __name__ == '__main__':
    x = tf.keras.layers.Input((256,))
    y = SpectralDecoder_RealFullFFT([256,256],modes=[8,10],out_channels=16,activation='relu')
    m = tf.keras.Sequential([x,y])
    m.summary()
        
        
        
        
        
