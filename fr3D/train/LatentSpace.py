import argparse
import json
import tensorflow as tf
import numpy as np
import h5py
import os
from tqdm import tqdm

from .utils import modify_node_types, prepare_dataset_for_training
from ..models.ConvAutoencoder import ConvAutoencoder
from ..models.ShallowDecoder import original_shallow_decoder
from ..data import DatasetPipelineBuilder
from ..data.utils import hdf5_train_test_split

tf.keras.backend.set_image_data_format('channels_last')

parser = argparse.ArgumentParser()
parser.add_argument('experiment_config', type=str)
parser.add_argument('dataset_path', type=str)
parser.add_argument('checkpoint_path', type=str)
parser.add_argument('--encoded_dataset', type=str, default=None)
parser.add_argument('--autoencoder_config_path', type=str, default=None)
parser.add_argument('--autoencoder_weights_path', type=str, default=None)
parser.add_argument('--load_weights', type=str, default=None)
parser.add_argument('--shuffle_size', type=int, default=500)
args = parser.parse_args()

config = json.load(open(args.experiment_config,'r'))

#load the autoencoder model and apply it to the dataset to create ground truths
if args.encoded_dataset is None or (not os.path.exists(args.encoded_dataset)):
    print('Encoding dataset...')
    autoencoder_config = json.load(open(args.autoencoder_config_path,'r'))
    encoder=None
    #feedforward_output_shape = tf.concat([[-1],tf.reduce_prod(encoder.output_shape[1:], keepdims=True)], 0)

    #setup datasets
    geometries = hdf5_train_test_split(args.dataset_path, [1.0], shuffle=False)[0]
    autoencoder_node_configs = modify_node_types(autoencoder_config['dataset']['node_configurations'], 'HDF5IODataset', 'filepath', args.dataset_path)

    if args.encoded_dataset is None:
        new_dataset_path = 'encoded_dataset.h5'
    else:
        new_dataset_path = args.encoded_dataset
    new_dataset = h5py.File(new_dataset_path, 'w')
    autoencoder_bsize = autoencoder_config['dataset']['batch_size']
    for geometry in tqdm(geometries):
        geometry_node_configurations = modify_node_types(autoencoder_node_configs, 'HDF5IODataset', 'groups', [geometry])
        geometry_dataset_pipeline = DatasetPipelineBuilder(geometry_node_configurations)
        geometry_dataset = geometry_dataset_pipeline.get_node(autoencoder_config['dataset']['training_node']).dataset.batch(autoencoder_bsize)
        geometry_results = []
        for batch in iter(geometry_dataset):
            if encoder is None:
                autoencoder_input_shape = batch.shape[1:]
                distribute_strategy =  tf.distribute.MirroredStrategy()
                with distribute_strategy.scope():
                    autoencoder_model = ConvAutoencoder(input_shape=autoencoder_input_shape, **autoencoder_config['model'])
                    autoencoder_model.load_weights(args.autoencoder_weights_path)
                    encoder = autoencoder_model.encoder
                    del autoencoder_model
            
            geometry_results.append(encoder(batch).numpy())
            
        geometry_results = np.concatenate(geometry_results,0)
        new_dataset.create_dataset(geometry, data=geometry_results)
    new_dataset.close()
    encoded_dataset = new_dataset_path
else:
    encoded_dataset = args.encoded_dataset

#load training and validation datasets   
sensor_node_configs = modify_node_types(config['sensor_dataset']['node_configurations'], 'HDF5IODataset', 'filepath', args.dataset_path)
encoding_node_configs = modify_node_types(config['encoding_dataset']['node_configurations'], 'HDF5IODataset', 'filepath', encoded_dataset)

if 'train_test_split' in config['training']:
    np.random.seed(42)
    train_geometries, test_geometries = hdf5_train_test_split(args.dataset_path, config['training']['train_test_split'][:2], shuffle=True)
    print(f'Training geometries: {train_geometries}')
    print(f'Test geometries: {test_geometries}')
    encoding_train_node_configurations = modify_node_types(encoding_node_configs, 'HDF5IODataset', 'groups', train_geometries)
    encoding_test_node_configurations = modify_node_types(encoding_node_configs, 'HDF5IODataset', 'groups', test_geometries)

    sensor_train_node_configurations = modify_node_types(sensor_node_configs, 'HDF5IODataset', 'groups', train_geometries)
    sensor_test_node_configurations = modify_node_types(sensor_node_configs, 'HDF5IODataset', 'groups', test_geometries)

    encoding_test_dataset_pipeline = DatasetPipelineBuilder(encoding_test_node_configurations)
    encoding_test_dataset = encoding_test_dataset_pipeline.get_node(config['encoding_dataset']['training_node']).dataset
    sensor_test_dataset_pipeline = DatasetPipelineBuilder(sensor_test_node_configurations)
    sensor_test_dataset = sensor_test_dataset_pipeline.get_node(config['sensor_dataset']['training_node']).dataset
    
    test_dataset = prepare_dataset_for_training(
        tf.data.Dataset.zip((sensor_test_dataset, encoding_test_dataset)),
        config['training']['batch_size'],
        args.shuffle_size)
    
encoding_train_dataset_pipeline = DatasetPipelineBuilder(encoding_train_node_configurations)
encoding_train_dataset = encoding_train_dataset_pipeline.get_node(config['encoding_dataset']['training_node']).dataset
sensor_train_dataset_pipeline = DatasetPipelineBuilder(sensor_train_node_configurations)
sensor_train_dataset = sensor_train_dataset_pipeline.get_node(config['sensor_dataset']['training_node']).dataset
train_dataset = prepare_dataset_for_training(
    tf.data.Dataset.zip((sensor_train_dataset, encoding_train_dataset)),
    config['training']['batch_size'],
    args.shuffle_size)

input_units = train_dataset.element_spec[0].shape[1]
output_units = train_dataset.element_spec[1].shape[1]

#model callbacks
callbacks = [tf.keras.callbacks.ModelCheckpoint(args.checkpoint_path, save_best_only=True, save_weights_only=True)]
if 'reduce_lr' in config['training']:
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(**config['training']['reduce_lr']))
if 'early_stopping' in config['training']:
    callbacks.append(tf.keras.callbacks.EarlyStopping(**config['training']['early_stopping']))

#create model and train
distribute_strategy =  tf.distribute.MirroredStrategy()
with distribute_strategy.scope():
    
    feedforward_input_shape = train_dataset.element_spec[0].shape[1]
    model = original_shallow_decoder(input_layer_shape=[input_units],
                                     output_layer_size=output_units,
                                     **config['model'])
    model.summary()
    loss_fn = tf.keras.losses.get(config['training']['loss'])
    model.compile(loss=loss_fn, optimizer = tf.keras.optimizers.get(config['training']['optimizer']), metrics = config['training'].get('metrics', None))

    if args.load_weights is not None:
        model.load_weights(args.load_weights)

    model.fit(train_dataset, epochs = config['training']['epochs'], callbacks = callbacks, validation_data = test_dataset, validation_steps = config['training'].get('validation_steps', None))
