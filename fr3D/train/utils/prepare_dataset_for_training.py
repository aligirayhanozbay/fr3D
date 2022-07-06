import tensorflow as tf

def prepare_dataset_for_training(ds, batch_size: int, shuffle_size: int = 1, prefetch_size: int = tf.data.AUTOTUNE):
    return ds.shuffle(shuffle_size).batch(batch_size).prefetch(prefetch_size)
