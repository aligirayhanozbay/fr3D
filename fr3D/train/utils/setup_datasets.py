import numpy as np

from .modify_node_types import modify_node_types
from .prepare_dataset_for_training import prepare_dataset_for_training
from ...data import DatasetPipelineBuilder
from ...data.utils import hdf5_train_test_split

def setup_datasets(config, dataset_path, shuffle_size=500):
    node_configs = modify_node_types(config['dataset']['node_configurations'], 'HDF5IODataset', 'filepath', dataset_path)

    if 'train_test_split' in config['dataset']:
        np.random.seed(42)
        train_geometries, test_geometries = hdf5_train_test_split(dataset_path, config['dataset']['train_test_split'][:2], shuffle=True)
        print(f'Training geometries: {train_geometries}')
        print(f'Test geometries: {test_geometries}')
        train_node_configurations = modify_node_types(node_configs, 'HDF5IODataset', 'groups', train_geometries)
        test_node_configurations = modify_node_types(node_configs, 'HDF5IODataset', 'groups', test_geometries)

        test_dataset_pipeline = DatasetPipelineBuilder(test_node_configurations)
        test_dataset = prepare_dataset_for_training(test_dataset_pipeline.get_node(config['dataset']['training_node']).dataset, config['dataset']['batch_size'], shuffle_size)
    else:
        train_node_configurations = node_configs
        test_dataset = None

    train_dataset_pipeline = DatasetPipelineBuilder(train_node_configurations)
    train_dataset = prepare_dataset_for_training(train_dataset_pipeline.get_node(config['dataset']['training_node']).dataset, config['dataset']['batch_size'], shuffle_size)

    return train_dataset, test_dataset
