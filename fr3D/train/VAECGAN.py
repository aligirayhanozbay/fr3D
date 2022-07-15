import argparse
import json
import tensorflow as tf
import numpy as np
import copy

from .utils import setup_datasets
from ..models import ConvVAE, VAECGAN

tf.keras.backend.set_image_data_format('channels_last')
#tf.keras.mixed_precision.set_global_policy('mixed_float16')

#parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('experiment_config', type=str)
parser.add_argument('dataset_path', type=str)
parser.add_argument('checkpoint_path', type=str)
parser.add_argument('--load_weights', type=str, default=None)
parser.add_argument('--generator_weights', type=str, default=None)
parser.add_argument('--shuffle_size', type=int, default=500)
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
#if 'reduce_lr' in config['training']:
#    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(**config['training']['reduce_lr']))
if 'early_stopping' in config['training']:
    callbacks.append(tf.keras.callbacks.EarlyStopping(**config['training']['early_stopping']))

distribute_strategy =  tf.distribute.MirroredStrategy()
with distribute_strategy.scope():
    generator = ConvVAE(input_shape=full_field_shape[1:], **config['generator'])
    generator.summary()

    if args.generator_weights is not None:
        generator.load_weights(args.generator_weights)

    GAN = VAECGAN(sensor_shape[1],
                  generator,
                  generator_output_shape=generator.decoder.output_shape[1:],
                  **config.get('discriminator',{}))
    GAN.compile(d_optimizer = tf.keras.optimizers.get(config['training']['d_optimizer']),
                g_optimizer = tf.keras.optimizers.get(config['training']['g_optimizer']),
                l_optimizer = tf.keras.optimizers.get(config['training']['l_optimizer']),
                global_batch_size=config['dataset']['batch_size'])
    GAN.discriminator.summary()
    GAN.latent_space_embedder.summary()

    if args.load_weights is not None:
        GAN.load_weights(args.load_weights)

    GAN.fit(train_dataset,
            epochs = config['training']['epochs'],
            callbacks = callbacks,
            validation_data = test_dataset,
            validation_steps = config['training'].get('validation_steps', None))
