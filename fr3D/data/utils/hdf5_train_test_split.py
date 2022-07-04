import numpy as np
import h5py

def hdf5_train_test_split(filepath: str, proportions: tuple[float], shuffle=True):
    '''
    Splits the top-level datasets/subgroups in a hdf5 file into chunks.
    The relative sizes of the chunks is defined by proportions.
    '''

    proportions = np.array(proportions) / np.sum(proportions)

    with h5py.File(filepath, 'r') as f:
            cases = list(f.keys())
    cases = np.sort(cases)#sorting necessary for reproducibility

    if shuffle:
        np.random.shuffle(cases)

    items_per_chunk = proportions*cases.shape[0]
    chunk_indices = [0] + [int(i) for i in np.cumsum(items_per_chunk[:-1])] + [cases.shape[0]-1]

    chunks = []
    for start_idx, end_idx in zip(chunk_indices[:-1], chunk_indices[1:]):
        chunks.append(tuple(cases[start_idx:end_idx]))

    return tuple(chunks)
    

    
