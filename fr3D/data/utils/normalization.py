import tensorflow as tf

from ...utils import get_all_subclasses

class BaseNormalization:
    name='base'
    def __init__(self, source='input', return_parameters=False, axis=-1, batch_mode=False):
        self.axis = axis
        self.source=source
        self.return_parameters=return_parameters
        self.batch_mode = batch_mode
        # self._call_func = self._make_call_function()

    def _get_parameters(self, x):
        raise(NotImplementedError())

    def _apply(self, x):
        raise(NotImplementedError())

    @tf.function
    def __call__(self, x, y):
        if self.source == 'input':
            norm_params_x = self._get_parameters(x)
            norm_params_y = norm_params_x
        elif self.source == 'target':
            norm_params_y = self._get_parameters(y)
            norm_params_x = norm_params_y
        elif self.source == 'individual':
            norm_params_x = self._get_parameters(x)
            norm_params_y = self._get_parameters(y)

        if self.batch_mode:
            xhat = tf.map_fn(self._apply, elems=(x, norm_params_x), fn_output_signature=x.dtype, parallel_iterations=10)
            yhat = tf.map_fn(self._apply, elems=(y, norm_params_y), fn_output_signature=y.dtype, parallel_iterations=10)
        else:
            xhat = self._apply((x, norm_params_x))
            yhat = self._apply((y, norm_params_y))
            
        if self.return_parameters:
            return xhat, yhat, norm_params_x, norm_params_y
        else:
            return xhat, yhat

    def undo(self, x, norm_params):
        raise(NotImplementedError())
        
class MeanCenterNormalization(BaseNormalization):
    name='meancenter'
    def _get_parameters(self, x):
        return tf.math.reduce_mean(x, axis=self.axis)

    def _apply(self, x):
        x, norm_params = x
        return x-norm_params

    def undo(self, xhat, norm_params):
        return xhat+norm_params

class MinMaxNormalization(BaseNormalization):
    name='minmax'
    def _get_parameters(self, x):
        xmin = tf.math.reduce_min(x, axis=self.axis)
        xmax = tf.math.reduce_max(x, axis=self.axis)
        return tf.stack([xmin, xmax], -1)

    def _apply(self, x):
        x, norm_params = x
        xmin = norm_params[...,0]
        xmax = norm_params[...,1]
        return (x-xmin)/(xmax - xmin)

    def undo(self, xhat, norm_params):
        xmin = norm_params[...,0]
        xmax = norm_params[...,1]
        return xhat*(xmax-xmin)+xmin
    
class ZScoreNormalization(BaseNormalization):
    name='zscore'
    
    def _get_parameters(self, x):
        xmean = tf.math.reduce_mean(x, axis=self.axis)
        xstd = tf.math.reduce_std(x, axis=self.axis)
        return tf.stack([xmean, xstd], -1)

    def _apply(self, x):
        x, norm_params = x
        mu = norm_params[...,0]
        std = norm_params[...,1]
        xhat = (x - mu)/std
        return xhat

    def undo(self, xhat, norm_params):
        mu = norm_params[...,0]
        std = norm_params[...,1]
        return xhat*std+mu

class UnitVectorNormalization(BaseNormalization):
    name='unitvector'
    def __init__(self, ord='euclidean', **kwargs):
        self.ord = ord
        super().__init__(**kwargs)
        
    def _get_parameters(self, x):
        return tf.linalg.norm(x, axis=self.axis, ord=self.ord)

    def _apply(self, x):
        x, norm_params = x
        return x/norm_params

    def undo(self, xhat, norm_params):
        return x*norm_params

def get_normalization(method, **kwargs):

    try:
        normalization_class = next(filter(lambda x: x.name == method, get_all_subclasses(BaseNormalization)))
    except:
        opts = [x.name for x in get_all_subclasses(BaseNormalization)]
        raise(RuntimeError(f'Could not find normalization type {method} - available options {opts}'))

    normalizer = normalization_class(**kwargs)

    return normalizer
    
if __name__ == '__main__':
    mu_inp = 2.0
    std_inp = 0.55
    out = tf.random.uniform((10,64,64,64,4))
    inp = tf.random.normal((10,256,4), mean = mu_inp, stddev = std_inp)

    normalization_classes = get_all_subclasses(BaseNormalization)
    normalizers = {n.name:get_normalization(n.name, axis=-2, return_parameters=True, batch_mode=True) for n in normalization_classes}
    
    m = {n:normalizers[n](inp, out) for n in normalizers}
    print({n:[x.shape for x in k] for n,k in zip(m.keys(), m.values())})

    out = tf.random.uniform((64,64,64,4))
    inp = tf.random.normal((256,4), mean = mu_inp, stddev = std_inp)

    normalization_classes = get_all_subclasses(BaseNormalization)
    normalizers = {n.name:get_normalization(n.name, axis=-2, return_parameters=True, batch_mode=False) for n in normalization_classes}
    
    m = {n:normalizers[n](inp, out) for n in normalizers}
    print({n:[x.shape for x in k] for n,k in zip(m.keys(), m.values())})
    
    
