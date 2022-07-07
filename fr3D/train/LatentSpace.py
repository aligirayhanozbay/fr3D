import argparse
import json
import tensorflow as tf
import numpy as np

from .utils import modify_node_types, prepare_dataset_for_training
from ..models.ConvAutoencoder import ConvAutoencoder
from ..models.ShallowDecoder import original_shallow_decoder
from ..data import DatasetPipelineBuilder
from ..data.utils import hdf5_train_test_split

tf.keras.backend.set_image_data_format('channels_last')

parser = argparse.ArgumentParser()
parser.add_argument('experiment_config', type=str)
parser.add_argument('dataset_path', type=str)
parser.add_argument('autoencoder_config_path', type=str)
parser.add_argument('autoencoder_weights_path', type=str)
parser.add_argument('checkpoint_path', type=str)
parser.add_argument('--load_weights', type=str, default=None)
parser.add_argument('--shuffle_size', type=int, default=500)
args = parser.parse_args()

config = json.load(open(args.experiment_config,'r'))
autoencoder_config = json.load(open(args.autoencoder_config_path,'r'))

#setup datasets
autoencoder_node_configs = modify_node_types(autoencoder_config['dataset']['node_configurations'], 'HDF5IODataset', 'filepath', args.dataset_path)
sensor_node_configs = modify_node_types(config['dataset']['node_configurations'], 'HDF5IODataset', 'filepath', args.dataset_path)

if 'train_test_split' in autoencoder_config['dataset']:
    np.random.seed(42)
    train_geometries, test_geometries = hdf5_train_test_split(args.dataset_path, autoencoder_config['dataset']['train_test_split'][:2], shuffle=True)
    print(f'Training geometries: {train_geometries}')
    print(f'Test geometries: {test_geometries}')
    autoencoder_train_node_configurations = modify_node_types(autoencoder_node_configs, 'HDF5IODataset', 'groups', train_geometries)
    autoencoder_test_node_configurations = modify_node_types(autoencoder_node_configs, 'HDF5IODataset', 'groups', test_geometries)

    sensor_train_node_configurations = modify_node_types(sensor_node_configs, 'HDF5IODataset', 'groups', train_geometries)
    sensor_test_node_configurations = modify_node_types(sensor_node_configs, 'HDF5IODataset', 'groups', test_geometries)

    autoencoder_test_dataset_pipeline = DatasetPipelineBuilder(autoencoder_test_node_configurations)
    autoencoder_test_dataset = autoencoder_test_dataset_pipeline.get_node(autoencoder_config['dataset']['training_node']).dataset
    sensor_test_dataset_pipeline = DatasetPipelineBuilder(sensor_test_node_configurations)
    sensor_test_dataset = sensor_test_dataset_pipeline.get_node(config['dataset']['training_node']).dataset
    test_dataset = prepare_dataset_for_training(
        tf.data.Dataset.zip((sensor_test_dataset, autoencoder_test_dataset)),
        config['dataset']['batch_size'],
        args.shuffle_size)
    
else:
    autoencoder_train_node_configurations = autoencoder_node_configs
    sensor_train_node_configurations = sensor_node_configs
    test_dataset = None

autoencoder_train_dataset_pipeline = DatasetPipelineBuilder(autoencoder_train_node_configurations)
autoencoder_train_dataset = autoencoder_train_dataset_pipeline.get_node(autoencoder_config['dataset']['training_node']).dataset
sensor_train_dataset_pipeline = DatasetPipelineBuilder(sensor_train_node_configurations)
sensor_train_dataset = sensor_train_dataset_pipeline.get_node(config['dataset']['training_node']).dataset
train_dataset = prepare_dataset_for_training(
    tf.data.Dataset.zip((sensor_train_dataset, autoencoder_train_dataset)),
    config['dataset']['batch_size'],
    args.shuffle_size)


autoencoder_dummy_input = next(iter(train_dataset))[1]
autoencoder_input_shape = autoencoder_dummy_input.shape[1:]
del autoencoder_dummy_input

#model callbacks
callbacks = [tf.keras.callbacks.ModelCheckpoint(args.checkpoint_path, save_best_only=True, save_weights_only=True)]
if 'reduce_lr' in config['training']:
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(**config['training']['reduce_lr']))
if 'early_stopping' in config['training']:
    callbacks.append(tf.keras.callbacks.EarlyStopping(**config['training']['early_stopping']))

distribute_strategy =  tf.distribute.MirroredStrategy()
with distribute_strategy.scope():
    #load the autoencoder model and apply it to the dataset to create ground truths
    autoencoder_model = ConvAutoencoder(input_shape=autoencoder_input_shape, **autoencoder_config['model'])
    autoencoder_model.load_weights(args.autoencoder_weights_path)
    encoder = autoencoder_model.encoder
    del autoencoder_model
    feedforward_output_shape = tf.concat([[-1],tf.reduce_prod(encoder.output_shape[1:], keepdims=True)], 0)

    @tf.function
    def encode_target(inp, target):
        encoding = encoder(target)
        return inp, tf.reshape(encoding, feedforward_output_shape)

    train_dataset = train_dataset.map(encode_target)
    if test_dataset is not None:
        test_dataset = test_dataset.map(encode_target)
        
    #create model and train
    feedforward_input_shape = train_dataset.element_spec[0].shape[1]
    model = original_shallow_decoder(input_layer_shape=[feedforward_input_shape],
                                     output_layer_size=int(feedforward_output_shape[1]),
                                     **config['model'])
    model.summary()
    loss_fn = tf.keras.losses.get(config['training']['loss'])
    model.compile(loss=loss_fn, optimizer = tf.keras.optimizers.get(config['training']['optimizer']), metrics = config['training'].get('metrics', None))

    if args.load_weights is not None:
        model.load_weights(args.load_weights)

    model.fit(train_dataset, epochs = config['training']['epochs'], callbacks = callbacks, validation_data = test_dataset, validation_steps = config['training'].get('validation_steps', None))
