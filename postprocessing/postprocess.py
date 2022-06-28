import re
import numpy
import os
import glob
import pathlib
import multiprocessing as mp
import json
import copy
import numpy as np
import h5py

from sample_native import process_case
from sampling_pts import get_samplingptshandler

def resolve_path(path):
    resolved_path = str(
        pathlib.Path(
            os.path.expanduser(path)
        ).resolve()
    )
    return resolved_path

class PostprocessingManager:
    time_window_match = re.compile('\d*[.,]?\d*.pyfrs$')
    verbose=True
    __doc__ ='''Inputs:
        -output_path: string. path to the outputted hdf5 file.
        -dataset_dir: string. path to the directory containing the solutions. the directory structure must be as follows:
        dataset_dir
        |
        |- case0
        |    |
        |    |- case0.pyfrm
        |    |- solution
        |         |-soln-0.00.pyfrs
        |         |-soln-0.05.pyfrs
        |         ...
        |         |-soln-{the final time}.pyfrs
        |- case1
        |    |
        |    ...
        ...

        -output_fields: tuple[str]. fields such as 'Pressure' and 'Velocity' which exist at the end of the paraview pipeline
        -sampling_pts_sensors: np.ndarray, shape (n_sensorpts, 3). sensor locations.
        -sampling_pts_fullfield: np.ndarray, shape (n_ffpts, 3). full field grid locations.
        -ignore_list: tuple[str]. list of cases to ignore
        -time_window: tuple[str,str]. solutions from times outside this range will be discarded.
        -save_dtype: str. "float32" or "float64".
        '''
    def __init__(self,
                 output_path: str,
                 dataset_dir: str,
                 sampling_pt_groups: dict,
                 ignore_list: tuple[str] = tuple(),
                 time_window = (-np.inf,np.inf),
                 save_dtype = 'float32'
                 ):
        
        self.output_path = resolve_path(output_path)
        self.save_dtype = save_dtype
        self.ignore_list = ignore_list

        files_list = self.get_solutions_and_meshes(resolve_path(dataset_dir), time_window=time_window)

        self.sampling_pt_groups = sampling_pt_groups

        self.read_queue = mp.Queue()
        self.write_queue = mp.Queue()

        self.soln_times_indices = {}
        for case_name, meshf, pyfrs_files in files_list:
            self.read_queue.put((case_name, meshf, pyfrs_files))
        self._nworkitems = self.read_queue.qsize()

    @staticmethod
    def _separate_sampling_results(x: np.ndarray, groups_info: dict, axis=1):

        #note: this line assumes entries in the dict are ordered. this requires recent versions of python
        separation_indices = [groups_info[spg]['npts'] for spg in groups_info]
        separation_indices = np.concatenate([[0], np.cumsum(separation_indices)],0)
    
        separated_result = {}
        for spg,start_idx,end_idx in zip(groups_info, separation_indices[:-1], separation_indices[1:]):
            # print(f'_separate_sampling_results: {spg} {start_idx} {end_idx}')
            # separated_result[spg] = x[start_idx:end_idx].reshape(*groups_info[spg]['orig_shape'],-1)
            sep_result = x.take(indices=range(start_idx, end_idx), axis=axis)
            old_shape = list(sep_result.shape)
            new_shape = old_shape[:axis] + list(groups_info[spg]['orig_shape']) + old_shape[axis+1:]
            separated_result[spg] = sep_result.reshape(*new_shape)    
        
        return separated_result
    
    def _get_soln_time_from_filename(self, x, extension='.pyfrs'):
        return float(self.time_window_match.findall(x)[-1][:-len(extension)])

    def get_solutions_and_meshes(self, path, time_window = (-np.inf,np.inf)):
        solns_and_meshes = []
        for directory in filter(lambda x: os.path.isdir(f'{path}/{x}') and all([ign not in x for ign in self.ignore_list]), os.listdir(path)):
            case_name = directory
            full_path_to_dir = f'{path}/{directory}'
            
            expected_mesh_file_name = f'{full_path_to_dir}/{case_name}.pyfrm'
            mesh_files_in_dir = glob.glob(f'{full_path_to_dir}/*.pyfrm')
            if expected_mesh_file_name in mesh_files_in_dir:
                meshf = expected_mesh_file_name
            else:
                meshf = mesh_files_in_dir[0]

            soln_files = glob.glob(f'{full_path_to_dir}/solution/*.pyfrs')
            soln_files_times = tuple(map(self._get_soln_time_from_filename, soln_files))
            
            if time_window != (-np.inf,np.inf):
                soln_files_mask = map(lambda x: x>=time_window[0] and x<=time_window[1], soln_files_times)
                soln_files, soln_files_times = tuple(zip(*map(lambda x: (x[0],x[2]), filter(lambda x: x[1], zip(soln_files, soln_files_mask, soln_files_times)))))

            soln_times_indices = {t:k for k,t in enumerate(soln_files_times)}
                
            solns_and_meshes.append((case_name, meshf, tuple(soln_files)))

            
        return tuple(solns_and_meshes)
            
    def run(self, n_readers=None):
        if n_readers is None:
            n_readers = max(mp.cpu_count()-4, self.read_queue.qsize())
        else:
            assert isinstance(n_readers, int) and n_readers>0
        
        self._nworkitems = self.read_queue.qsize()
        processes = [mp.Process(target=self.reader_process) for _ in range(n_readers)]
        writer_process = mp.Process(target=self.writer_process)
        for proc in processes:
            proc.start()
        writer_process.start()

        for proc in processes:
            proc.join()

        self.write_queue.put(-1)
        writer_process.join()
            
        return

    def reader_process(self):
        while not self.read_queue.empty():
            case_name, meshf, pyfrsf = self.read_queue.get()
            
            if self.verbose:
                rem_work_items = self.read_queue.qsize()
                print(f'Reading {case_name} ({self._nworkitems-rem_work_items}/{self._nworkitems})')

            sampling_pts = []
            groups_info = {}
            for sp in self.sampling_pt_groups:
                sampler_cfg = copy.deepcopy(self.sampling_pt_groups[sp])
                if 'pyfrm' in sampler_cfg['name']:
                    sampler_cfg['pyfrm'] = meshf
                sampler = get_samplingptshandler(**sampler_cfg)
                orig_shape = sampler.pts.shape[:-1]
                flattened_pts = sampler.pts.reshape(-1, sampler.pts.shape[-1])
                npts = flattened_pts.shape[0]
                sampling_pts.append(flattened_pts)
                groups_info[sp] = {'npts': npts, 'orig_shape': orig_shape}
            sampling_pts = np.concatenate(sampling_pts,0)
            
            pyfrsf = sorted(pyfrsf, key = self._get_soln_time_from_filename)
           
            sampling_result = process_case(meshf, pyfrsf, sampling_pts, verbose = self.verbose)
            sampling_result = self._separate_sampling_results(sampling_result, groups_info, axis=1)
            sampling_pts = self._separate_sampling_results(sampling_pts, groups_info, axis=0)
           
            self.write_queue.put((case_name, sampling_result, sampling_pts))
        return

    def writer_process(self):
        
        with h5py.File(self.output_path, 'w') as outf:
            while True:
                data = self.write_queue.get()
                if data == -1:
                    return

                case_name, sampling_result, sampling_pts = data
               
                sg = outf.create_group(case_name)
                for sampling_grp in sampling_result:
                    if self.verbose:
                        print(f'Writing {case_name} {sampling_result[sampling_grp].shape} {sampling_pts[sampling_grp].shape}')
                    sg.create_dataset(sampling_grp, data=sampling_result[sampling_grp], dtype=self.save_dtype)
                    sg.create_dataset(f'{sampling_grp}_coords', data=sampling_pts[sampling_grp], dtype=self.save_dtype)

                if self.verbose:
                    print(f'Wrote {case_name}')

def main():
    import argparse
    from sampling_pts import get_samplingptshandler

    explanation = '''
    Postprocess a folder of PyFR results, interpolating the results from each case to specified points.

    This script outputs a .h5 file containing two fields for each PyFR case, the 'sensors' and the 'full_field'.
    '''
    
    parser = argparse.ArgumentParser(explanation)
    parser.add_argument('input_dir', type=str, help='Folder containing the cases. See PostprocessingManager.__doc__')
    parser.add_argument('samplingconfig', type=str, help='JSON file containing the configurations for the sampling point objects')
    parser.add_argument('output_file', type=str, help='Path to the output file')
    parser.add_argument('-t', nargs=2, type=float, help='Time window. Provide 2 floats to mark tbegin and tend', default=(-np.inf,np.inf))
    parser.add_argument('--ignore', nargs='*', type=str, help='Cases (i.e. subdirectories) inside input_dir to ignore.', default=tuple())
    parser.add_argument('-d', type=str, help='Save datatype. float32 or float64.', default='float32')
    parser.add_argument('-np', type=int, default=None, help='Number of reader processes')

    args = parser.parse_args()

    sampling_conf = json.load(open(args.samplingconfig,'r'))

    manager = PostprocessingManager(args.output_file, args.input_dir, sampling_conf, ignore_list=args.ignore, save_dtype=args.d, time_window=args.t)
    
    manager.run(n_readers=args.np)

if __name__ == '__main__':
    main()
