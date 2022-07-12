import argparse
import json
import tensorflow as tf
import numpy as np
import copy

from .utils import setup_datasets
from ..models import ViTFR

tf.keras.backend.set_image_data_format('channels_last')
#tf.keras.mixed_precision.set_global_policy('mixed_float16')

#parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('experiment_config', type=str)
parser.add_argument('dataset_path', type=str)
parser.add_argument('checkpoint_path', type=str)
parser.add_argument('--load_weights', type=str, default=None)
parser.add_argument('--shuffle_size', type=int, default=500)
args = parser.parse_args()

config = json.load(open(args.experiment_config,'r'))

#setup datasets
#setup datasets
train_dataset, test_dataset = setup_datasets(config, args.dataset_path, args.shuffle_size)

input_shape = train_dataset.element_spec[0].shape
output_shape = train_dataset.element_spec[1].shape

#create model and train
callbacks = [tf.keras.callbacks.ModelCheckpoint(args.checkpoint_path,
                                                save_best_only=True,
                                                save_weights_only=True)]
if 'reduce_lr' in config['training']:
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(**config['training']['reduce_lr']))
if 'early_stopping' in config['training']:
    callbacks.append(tf.keras.callbacks.EarlyStopping(**config['training']['early_stopping']))


distribute_strategy =  tf.distribute.MirroredStrategy()
train_dataset = distribute_strategy.experimental_distribute_dataset(train_dataset)
if test_dataset is not None:
    test_dataset = distribute_strategy.experimental_distribute_dataset(test_dataset)
with distribute_strategy.scope():
    model = ViTFR(output_shape[1:], **config['model'])
    _ = model(next(iter(train_dataset)))
    model.summary()

    loss_fn = tf.keras.losses.get(config['training']['loss'])

    # if isinstance(config['training']['optimizer'], str):
    #     optim_config = {'class_name': config['training']['optimizer'], 'config': {}}
    # else:
    #     optim_config = copy.deepcopy(config['training']['optimizer'])
    #     if 'config' not in optim_config:
    #         optim_config['config'] = {}
    # base_lr = optim_config['config'].get('learning_rate', 1e-3)
    # optim_config['config']['learning_rate'] = WarmupCosineLRS(
    # optim = tf.keras.optimizers.get()
    
    model.compile(loss=loss_fn,
                  optimizer = tf.keras.optimizers.get(config['training']['optimizer']),
                  metrics = config['training'].get('metrics', None))

    if args.load_weights is not None:
        model.load_weights(args.load_weights)

    model.fit(train_dataset,
              epochs = config['training']['epochs'],
              callbacks = callbacks,
              validation_data = test_dataset,
              validation_steps = config['training'].get('validation_steps', None))
