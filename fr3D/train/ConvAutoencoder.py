import argparse
import json
import tensorflow as tf
import numpy as np

from .utils import setup_datasets
from ..models import ConvAutoencoder, ConvVAE


tf.keras.backend.set_image_data_format('channels_last')
#tf.keras.mixed_precision.set_global_policy('mixed_float16')

#parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('experiment_config', type=str)
parser.add_argument('dataset_path', type=str)
parser.add_argument('checkpoint_path', type=str)
parser.add_argument('--load_weights', type=str, default=None)
parser.add_argument('--shuffle_size', type=int, default=500)
parser.add_argument('--vae', action='store_true')
args = parser.parse_args()

config = json.load(open(args.experiment_config,'r'))

#setup datasets
train_dataset, test_dataset = setup_datasets(config, args.dataset_path, args.shuffle_size)

input_shape = train_dataset.element_spec.shape

#create model and train
callbacks = [tf.keras.callbacks.ModelCheckpoint(args.checkpoint_path, save_weights_only=True, **config['training']['model_checkpoint'])]
if 'reduce_lr' in config['training']:
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(**config['training']['reduce_lr']))
if 'early_stopping' in config['training']:
    callbacks.append(tf.keras.callbacks.EarlyStopping(**config['training']['early_stopping']))

distribute_strategy =  tf.distribute.MirroredStrategy()
with distribute_strategy.scope():
    if args.vae:
        model = ConvVAE(input_shape=input_shape, **config['model'])
    else:
        model = ConvAutoencoder(input_shape=input_shape, **config['model'])
    model.summary()
    loss_fn = tf.keras.losses.get(config['training']['loss'])
    model.compile(loss=loss_fn, optimizer = tf.keras.optimizers.get(config['training']['optimizer']), metrics = config['training'].get('metrics', None))

    if args.load_weights is not None:
        model.load_weights(args.load_weights)

    model.fit(train_dataset, epochs = config['training']['epochs'], callbacks = callbacks, validation_data = test_dataset, validation_steps = config['training'].get('validation_steps', None))
