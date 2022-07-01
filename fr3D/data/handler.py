import copy
import h5py
import tensorflow as tf
import tensorflow_io as tfio

from .utils import get_normalization

def merge_datasets(datasets: list):
    ds = datasets[0]
    for d in datasets[1:]:
        ds = ds.concatenate(d)
    return ds


saved_var_order = ('p', 'u', 'v', 'w', 'p_x', 'p_y', 'p_z',
                   'u_x', 'u_y', 'u_z', 'v_x', 'v_y', 'v_z',
                   'w_x', 'w_y', 'w_z')
reverse_saved_var_order = {v:k for k,v in enumerate(saved_var_order)}

def keep_vars(ds, keep_vars):
    if isinstance(keep_vars, list) or isinstance(keep_vars, tuple):
            keep_vars = {'axis': -1, 'indices': [reverse_saved_var_order[v] if isinstance(v, str) else v for v in keep_vars]}
    return ds.map(lambda x: tf.gather(x, **keep_vars))


def reshape_data(ds, reshape):
    reshape = [len(saved_var_order) if v=='N_VARS' else v for v in reshape]
    return ds.map(lambda x: tf.reshape(x, reshape))


def normalize_data(ds, normalization_spec):
    normalization_spec['batch_mode']=False
    normalizer = get_normalization(**normalization_spec)
    return ds.map(normalizer)


def loadIODataset(filepath, configs, concat_axis: int = -1, reshape: tuple = None, keep_vars = None):
    if isinstance(configs, dict):
        configs = [configs]
        
    if len(configs) == 1:
        config = copy.deepcopy(configs[0])
        init_options = config.get('init_options', {})
        
        with h5py.File(filepath, 'r') as f:
            cases = list(f.keys())

        ds = merge_datasets([tfio.IODataset.from_hdf5(filepath, f'/{case}/{config["field"]}', **init_options) for case in cases])

    else:
        datasets = tuple([loadIODataset(filepath, **cfg) for cfg in configs])
        ds = tf.data.Dataset.zip(datasets).map(lambda *x: tf.concat(x, axis=concat_axis))

    if keep_vars is not None:
        pass

    if reshape is not None:
        pass

    return ds

def Dataset(filepath, configs: tuple):
    datasets = []
    for cfgs in configs:
        datasets.append(loadIODataset(filepath, **cfgs))

    ds = tf.data.Dataset.zip(tuple(datasets))
        
    return ds
    

if __name__ == '__main__':
    
    fpath = '/fr3D/postprocessed/postprocessed_test.h5'
    configs = [
        {
            "configs": [
                {"configs": {"field": "sensors-velocity"}, "reshape": [-1], "keep_vars": ["u", "v", "w"]},
                {"configs": {"field": "sensors-pressure"}, "reshape": [-1], "keep_vars": ["p"]}
            ],
            "concat_axis": -1,
        },
        {
            "configs": {"field": "full_field"},
            "keep_vars": ["u","w"]
        }
    ]

    #ds = loadIODataset(fpath, configs)
    datasets = Dataset(fpath, configs)
    import pdb; pdb.set_trace()

    
