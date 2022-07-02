import tensorflow as tf

from ...utils import get_all_subclasses

class BaseNormalization:
    name='base'
    def __init__(self, source='input', return_parameters=False, axis=-1, batch_mode=False):
        self.axis = axis
        self.source=source
        self.return_parameters=return_parameters
        self.batch_mode = batch_mode
        self._call_func = self._make_call_function()

    def _get_parameters(self, x):
        raise(NotImplementedError())

    def _apply(self, x):
        raise(NotImplementedError())

    def _make_call_function(self):
        if self.source == 'input':
            get_parameters_wrap = lambda x, y: (self._get_parameters(x), self._get_parameters(x))
        elif self.source == 'target':
            get_parameters_wrap = lambda x, y: (self._get_parameters(y), self._get_parameters(y))
        elif self.source == 'individual':
            get_parameters_wrap = lambda x, y: (self._get_parameters(x), self._get_parameters(y))

        if self.batch_mode:
            apply_wrap = lambda x, norm_params: tf.map_fn(self._apply, elems=(x, norm_params), fn_output_signature=x.dtype, parallel_iterations=10)
        else:
            apply_wrap = lambda x, norm_params: self._apply((x, norm_params))

        if self.return_parameters:
            pack_parameters = lambda *x: x
        else:
            pack_parameters = lambda *x: (x[0], x[1])

        @tf.function
        def call(sample):
            x,y = sample
            norm_params_x, norm_params_y = get_parameters_wrap(x, y)

            xhat = apply_wrap(x, norm_params_x)
            yhat = apply_wrap(y, norm_params_y)

            return pack_parameters(xhat, yhat, norm_params_x, norm_params_y)

        return call

    
    def __call__(self, x, y):
        return self._call_func(x,y)

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
    
    
