import argparse
import json
import tensorflow as tf
import numpy as np

from .utils import modify_node_types, prepare_dataset_for_training
from ..models import ConvAutoencoder, ConvVAE
from ..data import DatasetPipelineBuilder
from ..data.utils import hdf5_train_test_split

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
node_configs = modify_node_types(config['dataset']['node_configurations'], 'HDF5IODataset', 'filepath', args.dataset_path)

if 'train_test_split' in config['dataset']:
    np.random.seed(42)
    train_geometries, test_geometries = hdf5_train_test_split(args.dataset_path, config['dataset']['train_test_split'][:2], shuffle=True)
    print(f'Training geometries: {train_geometries}')
    print(f'Test geometries: {test_geometries}')
    train_node_configurations = modify_node_types(node_configs, 'HDF5IODataset', 'groups', train_geometries)
    test_node_configurations = modify_node_types(node_configs, 'HDF5IODataset', 'groups', test_geometries)

    test_dataset_pipeline = DatasetPipelineBuilder(test_node_configurations)
    test_dataset = prepare_dataset_for_training(test_dataset_pipeline.get_node(config['dataset']['training_node']).dataset, config['dataset']['batch_size'], args.shuffle_size)
else:
    train_node_configurations = node_configs
    test_dataset = None

train_dataset_pipeline = DatasetPipelineBuilder(train_node_configurations)
train_dataset = prepare_dataset_for_training(train_dataset_pipeline.get_node(config['dataset']['training_node']).dataset, config['dataset']['batch_size'], args.shuffle_size)

dummy_input = next(iter(train_dataset))
input_shape = dummy_input.shape[1:]
del dummy_input

#create model and train
callbacks = [tf.keras.callbacks.ModelCheckpoint(args.checkpoint_path, save_best_only=True, save_weights_only=True)]
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
