import copy
import h5py
import tensorflow as tf
import tensorflow_io as tfio

from .utils import get_normalization
from ..utils import get_all_subclasses

def merge_datasets(datasets: list):
    ds = datasets[0]
    for d in datasets[1:]:
        ds = ds.concatenate(d)
    return ds    

class DatasetPipelineNode:
    nodetype='base'
    n_inputnodes=0
    def __init__(self, identifier: str, inputs: tuple[str], nodes_in_graph: dict={}):
        self.nodes_in_graph = nodes_in_graph
        self.identifier = identifier
        self.nodes_in_graph[self.identifier] = self

        self.inputs = inputs
        if isinstance(self.n_inputnodes, int):
            assert len(self.inputs) == self.n_inputnodes
        elif self.n_inputnodes == 'n':
            assert len(self.inputs) >= 0
        
        self._dataset = None

    def _transform(self):
        raise(NotImplementedError())

    @property
    def dataset(self):
        if self._dataset is None:
            input_datasets = []
            for node_name in self.inputs:
                input_dataset = self.nodes_in_graph[node_name].dataset
                input_datasets.append(input_dataset)
            self._dataset = self._transform(*input_datasets)
        return self._dataset

    @classmethod
    def make_node(cls, nodetype, **kwargs):
        try:
            node_class = next(filter(lambda x: x.nodetype == nodetype, get_all_subclasses(cls)))
        except:
            opts = [x.nodetype for x in get_all_subclasses(cls)]
            raise(RuntimeError(f'Could not find dataset pipeline node type {nodetype} - available options {opts}'))
        pipeline_node = node_class(**kwargs)
        return pipeline_node
                    
class KeepVarsNode(DatasetPipelineNode):
    saved_var_order = ('p', 'u', 'v', 'w', 'p_x', 'p_y', 'p_z',
                   'u_x', 'u_y', 'u_z', 'v_x', 'v_y', 'v_z',
                   'w_x', 'w_y', 'w_z')
    
    nodetype='keep_vars'
    n_inputnodes=1
    def __init__(self, vars_to_keep, axis=-1, **kwargs):
        self.vars_to_keep =  tuple([(self.reverse_saved_var_order)[v] if isinstance(v, str) else v for v in vars_to_keep])
        self.axis = axis
        super().__init__(**kwargs)

    @classmethod
    @property
    def reverse_saved_var_order(cls):
        return {v:k for k,v in enumerate(cls.saved_var_order)}

    def _transform(self, ds):
        return ds.map(lambda x: tf.gather(x, indices=self.vars_to_keep, axis=self.axis))

class HDF5IODatasetNode(DatasetPipelineNode):
    nodetype='HDF5IODataset'
    n_inputnodes=0
    def __init__(self, filepath: str, field: str, identifier: str, nodes_in_graph: dict[str, DatasetPipelineNode]={}, groups: tuple[str] = None, **tfio_IODataset_options):

        super().__init__(identifier=identifier, inputs=tuple(), nodes_in_graph=nodes_in_graph)

        if groups is None:
            with h5py.File(filepath, 'r') as f:
                groups = list(f.keys())
            
        self._dataset = merge_datasets([tfio.IODataset.from_hdf5(filepath, f'/{group}/{field}', **tfio_IODataset_options) for group in groups])
    
class TakeElementNode(DatasetPipelineNode):
    nodetype='take'
    n_inputnodes=1
    def __init__(self, take_idx, **kwargs):
        self.take_idx = take_idx
        super().__init__(**kwargs)
        
    def _transform(self, ds):
        return ds.map(lambda *x: x[self.take_idx])

class ReshapeNode(DatasetPipelineNode):
    nodetype='reshape'
    n_inputnodes=1
    def __init__(self, new_shape, **kwargs):
        self.new_shape = new_shape
        super().__init__(**kwargs)

    def _transform(self, ds):
        return ds.map(lambda x: tf.reshape(x, self.new_shape))

class ZipNode(DatasetPipelineNode):
    nodetype='zip'
    n_inputnodes='n'
    def _transform(self, *ds):
        return tf.data.Dataset.zip(tuple(ds))
        
class NormalizeNode(DatasetPipelineNode):
    nodetype='normalize'
    n_inputnodes=1
    def __init__(self, normalization_spec: dict, **kwargs):
        normalization_spec = copy.deepcopy(normalization_spec)
        normalization_spec['batch_mode'] = False
        self.normalizer = get_normalization(**normalization_spec)
        super().__init__(**kwargs)

    def _transform(self, ds):
        return ds.map(self.normalizer)

class ConcatenateNode(DatasetPipelineNode):
    nodetype='concatenate'
    n_inputnodes=1
    def __init__(self, axis=-1, **kwargs):
        self.axis=axis
        super().__init__(**kwargs)

    def _transform(self, ds):
        return ds.map(lambda *x: tf.concat(x, axis=self.axis))


class DatasetPipelineBuilder:
    def __init__(self, node_configurations: tuple):
        self.nodes_in_graph = {}
        for node_conf in node_configurations:
            node = self.add_node(node_conf)

    def add_node(self, node_conf):
        node = DatasetPipelineNode.make_node(**node_conf, nodes_in_graph=self.nodes_in_graph)
            
    def get_node(self, name:str):
        return self.nodes_in_graph[name]
    
    

if __name__ == '__main__':
    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()

    
    fpath = '/fr3D/postprocessed/postprocessed_test.h5'
    node_configurations = [
        {
            "nodetype": "HDF5IODataset",
            "identifier": "sensors-pressure",
            "filepath": fpath,
            "field": "sensors-pressure"
        },
        {
            "nodetype": "HDF5IODataset",
            "identifier": "sensors-velocity",
            "filepath": fpath,
            "field": "sensors-velocity"
        },
        {
            "nodetype": "HDF5IODataset",
            "identifier": "full_field",
            "filepath": fpath,
            "field": "full_field"
        },
        {
            "nodetype": "keep_vars",
            "identifier": "sensors-pressure-keepvars",
            "vars_to_keep": ["p"],
            "inputs": ["sensors-pressure"]
        },
        {
            "nodetype": "keep_vars",
            "identifier": "sensors-velocity-keepvars",
            "vars_to_keep": ["u", "v", "w"],
            "inputs": ["sensors-velocity"]
        },
        {
            "nodetype": "keep_vars",
            "identifier": "full_field_pressure",
            "vars_to_keep": ["p"],
            "inputs": ["full_field"]
        },
        {
            "nodetype": "keep_vars",
            "identifier": "full_field_velocity",
            "vars_to_keep": ["u", "v", "w"],
            "inputs": ["full_field"]
        },
        {
            "nodetype": "zip",
            "identifier": "zip_pressure",
            "inputs": ["sensors-pressure-keepvars", "full_field_pressure"]
        },
        {
            "nodetype": "zip",
            "identifier": "zip_velocity",
            "inputs": ["sensors-velocity-keepvars", "full_field_velocity"]
        },
        {
            "nodetype": "normalize",
            "identifier": "normalize_pressure",
            "inputs": ["zip_pressure"],
            "normalization_spec": {
                "method": "zscore",
                "axis": [-2,-3,-4],
                "return_parameters": True
            }
        },
        {
            "nodetype": "normalize",
            "identifier": "normalize_velocity",
            "inputs": ["zip_velocity"],
            "normalization_spec": {
                "method": "zscore",
                "axis": [-2,-3,-4],
                "return_parameters": True
            }
        },
        {
            "nodetype": "take",
            "identifier": "pressure_input",
            "take_idx": 0,
            "inputs": ["normalize_pressure"]
        },
        {
            "nodetype": "take",
            "identifier": "pressure_output",
            "take_idx": 1,
            "inputs": ["normalize_pressure"]
        },
        {
            "nodetype": "take",
            "identifier": "velocity_input",
            "take_idx": 0,
            "inputs": ["normalize_velocity"]
        },
        {
            "nodetype": "take",
            "identifier": "velocity_output",
            "take_idx": 1,
            "inputs": ["normalize_velocity"]
        },
        {
            "nodetype": "zip",
            "identifier": "zip_output",
            "inputs": ["pressure_output", "velocity_output"]
        },
        {
            "nodetype": "concatenate",
            "identifier": "concatenate_output",
            "inputs": ["zip_output"],
            "axis": -1
        },
        {
            "nodetype": "keep_vars",
            "identifier": "output",
            "inputs": ["concatenate_output"],
            "vars_to_keep": ["p"]
        },
        {
            "nodetype": "reshape",
            "identifier": "reshape_pressure_input",
            "inputs": ["pressure_input"],
            "new_shape": [-1]
        },
        {
            "nodetype": "reshape",
            "identifier": "reshape_velocity_input",
            "inputs": ["velocity_input"],
            "new_shape": [-1]
        },
        {
            "nodetype": "zip",
            "identifier": "zip_input",
            "inputs": ["reshape_pressure_input", "reshape_velocity_input"],
        },
        {
            "nodetype": "concatenate",
            "identifier": "input",
            "inputs": ["zip_input"],
            "axis": -1
        }
    ]
    dsp = DatasetPipelineBuilder(node_configurations)
    import pdb; pdb.set_trace()
    z=1

    
