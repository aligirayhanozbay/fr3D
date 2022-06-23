import re
import numpy
import os
import glob
import pathlib
import multiprocessing as mp
import numpy as np
import h5py

from sample_native import process_case

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
                 sampling_pts_sensors: np.ndarray,
                 sampling_pts_fullfield: np.ndarray,
                 ignore_list: tuple[str] = tuple(),
                 time_window = (-np.inf,np.inf),
                 save_dtype = 'float32'
                 ):
        
        self.output_path = resolve_path(output_path)
        self.save_dtype = save_dtype
        self.ignore_list = ignore_list

        files_list = self.get_solutions_and_meshes(resolve_path(dataset_dir), time_window=time_window)

        self.sampling_pts_sensors_n = sampling_pts_sensors.reshape(-1,sampling_pts_sensors.shape[-1]).shape[0]
        self.sampling_pts_sensors_shape = sampling_pts_sensors.shape[:-1]
        self.sampling_pts_fullfield_n = sampling_pts_fullfield.reshape(-1,sampling_pts_fullfield.shape[-1]).shape[0]
        self.sampling_pts_fullfield_shape = sampling_pts_fullfield.shape[:-1]
        self.sampling_pts = np.concatenate([sampling_pts_sensors.reshape(-1,sampling_pts_sensors.shape[-1]), sampling_pts_fullfield.reshape(-1,sampling_pts_fullfield.shape[-1])],0)

        self.read_queue = mp.Queue()
        self.write_queue = mp.Queue()

        self.soln_times_indices = {}
        for case_name, meshf, pyfrs_files in files_list:
            self.read_queue.put((case_name, meshf, pyfrs_files))

    
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
            
    def run(self):
        n_readers = max(mp.cpu_count(), self.read_queue.qsize())
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
            
            pyfrsf = sorted(pyfrsf, key = self._get_soln_time_from_filename)
            
            sampling_result = process_case(meshf, pyfrsf, self.sampling_pts)
            sensor_values = sampling_result[:,:self.sampling_pts_sensors_n].reshape(sampling_result.shape[0], *self.sampling_pts_sensors_shape, -1)
            full_field_values = sampling_result[:,self.sampling_pts_sensors_n:].reshape(sampling_result.shape[0], *self.sampling_pts_fullfield_shape, -1)
            
            self.write_queue.put((case_name, sensor_values, full_field_values))
        return

    def writer_process(self):
        with h5py.File(self.output_path, 'w') as outf:
           while True:
               data = self.write_queue.get()
               if data == -1:
                   return

               case_name, sensor_values, full_field_values = data
               
               sg = outf.create_group(case_name)
               sg.create_dataset('sensors', data=sensor_values, dtype=self.save_dtype)
               sg.create_dataset('full_field', data=full_field_values, dtype=self.save_dtype)

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
    parser.add_argument('output_file', type=str, help='Path to the output file')
    parser.add_argument('-t', nargs=2, type=float, help='Time window. Provide 2 floats to mark tbegin and tend', default=(-np.inf,np.inf))
    parser.add_argument('--ignore', nargs='*', type=str, help='Cases (i.e. subdirectories) inside input_dir to ignore.', default=tuple())
    parser.add_argument('-d', type=str, help='Save datatype. float32 or float64.', default='float32')

    args = parser.parse_args()
    

if __name__ == '__main__':
    c = np.stack(np.meshgrid(
        np.linspace(2.0,4.0,16),
        np.linspace(-1.0,1.0,16),
        np.linspace(2.0,4.0,16),
        indexing='ij'
    ),-1)
    nsensors = 128
    s = np.stack([5*np.ones(nsensors),0*np.ones(nsensors),np.linspace(2.0,8.0,nsensors)],-1)

    pm = PostprocessingManager('/tmp/test.h5', '/fr3D/pp_test2',
                               sampling_pts_sensors=s,
                               sampling_pts_fullfield=c,
                               time_window=[20.0,60.0]
                               )
    pm.run()
