import argparse
import json
import tensorflow as tf
import numpy as np
import copy

from .utils import setup_datasets
from ..models import ConvAutoencoderC

tf.keras.backend.set_image_data_format('channels_last')
#tf.keras.mixed_precision.set_global_policy('mixed_float16')

#parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('experiment_config', type=str)
parser.add_argument('dataset_path', type=str)
parser.add_argument('checkpoint_path', type=str)
parser.add_argument('--load_weights', type=str, default=None)
parser.add_argument('--shuffle_size', type=int, default=1)
args = parser.parse_args()

config = json.load(open(args.experiment_config,'r'))

#setup datasets
train_dataset, test_dataset = setup_datasets(config, args.dataset_path, args.shuffle_size)

sensor_shape = train_dataset.element_spec[0].shape
full_field_shape = train_dataset.element_spec[1].shape

#create model and train
callbacks = [tf.keras.callbacks.ModelCheckpoint(args.checkpoint_path,
                                                save_weights_only=True,
                                                **config['training']['model_checkpoint'])]
# if 'reduce_lr' in config['training']:
#    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(**config['training']['reduce_lr']))
if 'early_stopping' in config['training']:
    callbacks.append(tf.keras.callbacks.EarlyStopping(**config['training']['early_stopping']))

distribute_strategy =  tf.distribute.MirroredStrategy()
with distribute_strategy.scope():
    model = ConvAutoencoderC(dense_input_units=sensor_shape[1],
                             autoencoder_input_shape=full_field_shape[1:],
                             **config['model'])
    model.summary()

    if len(config['training']['loss']) == 3 and config['training']['loss'][:2] == "Lp":
        p = int(config['training']['loss'][-1])
        
        @tf.function
        def Lp_loss(yt,yp):
            if p%2 == 0:
                return tf.reduce_mean((yt-yp)**p)
            else:
                return tf.reduce_mean(tf.abs((yt-yp)**p))
        
        loss_fn = Lp_loss
    else:
        loss_fn = tf.keras.losses.get(config['training']['loss'])
    
    model.compile(l_optimizer= tf.keras.optimizers.get(config['training']['l_optimizer']),
                  loss=loss_fn,
                  optimizer = tf.keras.optimizers.get(config['training']['ae_optimizer']),
                  metrics = config['training'].get('metrics', None),
                  latent_space_step_ratio = config['training'].get('latent_space_step_ratio', 1))

    if args.load_weights is not None:
        model.load_weights(args.load_weights)

    model.fit(train_dataset, epochs = config['training']['epochs'], callbacks = callbacks, validation_data = test_dataset, validation_steps = config['training'].get('validation_steps', None))
