import tensorflow as tf
import numpy as np

@tf.function
def mape_with_threshold(yp, yt, pcterror_threshold=np.inf, max_magnitude_threshold=0.0, eps=1e-7, sample_weights=None, axis=None):
    if sample_weights is None:
        sample_weights = 1.0
        
    pct_errors = 100*tf.abs((yp-yt)/(eps + yt))
    pcterror_mask = pct_errors < pcterror_threshold
    max_magnitude_mask = tf.logical_not(tf.abs(yt) < (max_magnitude_threshold*tf.reduce_max(tf.abs(yt))))
    cond_mask = tf.logical_and(pcterror_mask, max_magnitude_mask)
    
    sample_weights = sample_weights * tf.cast(cond_mask, pct_errors.dtype)
    
    return tf.reduce_sum(sample_weights*pct_errors, axis=axis)/tf.reduce_sum(sample_weights, axis=axis)
