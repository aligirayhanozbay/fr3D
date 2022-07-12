import argparse
import json
import tensorflow as tf
import numpy as np
import copy

from .utils import setup_datasets
from ..models import ViTFR, ViTGAN


distribute_strategy =  tf.distribute.MultiWorkerMirroredStrategy()
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

input_shape = train_dataset.element_spec[0].shape
output_shape = train_dataset.element_spec[1].shape

#create model and train
callbacks = [tf.keras.callbacks.ModelCheckpoint(args.checkpoint_path,
                                                save_best_only=False,
                                                save_weights_only=True)]


with distribute_strategy.scope():
    generator = ViTFR(output_shape[1:], **config['generator'])
    _ = generator(next(iter(train_dataset)))
    generator.summary()

    if args.generator_weights is not None:
        generator.load_weights(args.generator_weights)

    recons_loss = tf.keras.losses.MeanSquaredError(reduction='none')
    GAN = ViTGAN(generator,
                 generator_output_shape=output_shape[1:],
                 **config.get('discriminator',{}))
    
    GAN.compile(d_optimizer = tf.keras.optimizers.get(config['training']['d_optimizer']),
                g_optimizer = tf.keras.optimizers.get(config['training']['g_optimizer']),
                global_batch_size=config['dataset']['batch_size'])
    GAN.discriminator.summary()

    if args.load_weights is not None:
        GAN.load_weights(args.load_weights)

    GAN.fit(train_dataset,
            epochs = config['training']['epochs'],
            callbacks = callbacks,
            validation_data = test_dataset,
            validation_steps = config['training'].get('validation_steps', None))
