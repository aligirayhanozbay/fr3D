import warnings
import numpy as np

from .modify_node_types import modify_node_types
from .prepare_dataset_for_training import prepare_dataset_for_training
from ...data import DatasetPipelineBuilder
from ...data.utils import hdf5_train_test_split

def add_case_name_output(node_configs: list[dict], output_node: str):
    #copy some parameters from a HDF5IODataset node
    ionode=None
    for node in node_configs:
        if node['nodetype'] == "HDF5IODataset":
            ionode = node
            break
    if ionode == None:
        raise(RuntimeError('Could not find a HDF5IODataset node to copy config from for outputting case names'))
    else:
        groups = node.get('groups', None)
        field = node.get('field', None)
        dataset_path = node['filepath']

    case_names_node_cfg = {"nodetype": "HDF5DataPath",
                           "identifier": "case_names",
                           "filepath": dataset_path,
                           "groups": groups,
                           "field": field}
    zip_node_cfg = {"nodetype": "zip",
                    "identifier": output_node + "_case_name_zip",
                    "inputs": [output_node, "case_names"]}

    return list(node_configs) + [case_names_node_cfg, zip_node_cfg]
        
    

def setup_datasets(config, dataset_path, shuffle_size=500, case_names=False, evaluation=False):
    node_configs = modify_node_types(config['dataset']['node_configurations'], 'HDF5IODataset', 'filepath', dataset_path)

    if evaluation:
        try:
            orig_output_node = config['dataset']['evaluation_node']
        except:
            orig_output_node = config['dataset']['training_node']
            warnings.warn("config['dataset']['evaluation_node'] missing - using config['dataset']['training_node'] instead")
    else:
        orig_output_node = config['dataset']['training_node']

    if case_names:
        output_node = orig_output_node + '_case_name_zip'
    else:
        output_node = orig_output_node

    if 'train_test_split' in config['dataset']:
        np.random.seed(42)
        train_geometries, test_geometries = hdf5_train_test_split(dataset_path, config['dataset']['train_test_split'][:2], shuffle=True)
        
        print(f'Training geometries: {train_geometries}')
        print(f'Test geometries: {test_geometries}')
        train_node_configurations = modify_node_types(node_configs, 'HDF5IODataset', 'groups', train_geometries)
        test_node_configurations = modify_node_types(node_configs, 'HDF5IODataset', 'groups', test_geometries)

        if case_names:
            test_node_configurations = add_case_name_output(test_node_configurations, orig_output_node)

        test_dataset_pipeline = DatasetPipelineBuilder(test_node_configurations)
        test_dataset = prepare_dataset_for_training(test_dataset_pipeline.get_node(output_node).dataset, config['dataset']['batch_size'], shuffle_size)
    else:
        train_node_configurations = node_configs
        test_dataset = None

        
    if case_names:
        train_node_configurations = add_case_name_output(train_node_configurations, orig_output_node)
    train_dataset_pipeline = DatasetPipelineBuilder(train_node_configurations)
    train_dataset = prepare_dataset_for_training(train_dataset_pipeline.get_node(output_node).dataset, config['dataset']['batch_size'], shuffle_size)

    return train_dataset, test_dataset
