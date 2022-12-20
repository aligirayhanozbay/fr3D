import tensorflow as tf
import numpy as np

@tf.function
def mape_with_threshold(yp, yt, pcterror_threshold=np.inf, max_magnitude_threshold=0.0, eps=1e-7):
    pct_errors = 100*tf.abs((yp-yt)/(eps + yt))
    pcterror_mask = pct_errors < pcterror_threshold
    max_magnitude_mask = tf.logical_not(tf.abs(yt) < (max_magnitude_threshold*tf.reduce_max(tf.abs(yt))))
    filtering_indices = tf.where(tf.logical_and(pcterror_mask, max_magnitude_mask))
    filtered_pcterrors = tf.gather_nd(pct_errors, filtering_indices)
    return tf.reduce_mean(filtered_pcterrors)
