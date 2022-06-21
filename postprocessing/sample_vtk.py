import pathlib
import os
import secrets
import numpy as np
import tempfile
from dataclasses import dataclass
import time

from paraview.simple import OpenDataFile, TableToPoints, CSVReader, ResampleWithDataset, PassArrays, Delete
from paraview import servermanager as sm
from paraview.vtk.numpy_interface import dataset_adapter as dsa

from pyfr.writers.vtk import VTKWriter as PyFRVTKWriter

def resolve_path(path):
    resolved_path = str(
        pathlib.Path(
            os.path.expanduser(path)
        ).resolve()
    )
    return resolved_path

def sample_vtk(filename: str, sampling_pts: np.ndarray, output_fields: tuple[str], filters: tuple = tuple()):

    #handle filename
    fpath = resolve_path(filename)

    #load file and run pipeline
    t0 = time.time()
    pvobj = OpenDataFile(fpath)
    pipeline = [pvobj]
    for filt in filters:
        pvobj = filt[0](pvobj, **filt[1])
        pipeline.append(pvobj)
    pvobj = PassArrays(pvobj, PointDataArrays=output_fields)
    pipeline.append(pvobj)
    pipeline[-1].UpdatePipeline()
    t1 = time.time()
    print(f'Primary pipeline: {t1-t0} secs')

    #make a temporary csv file for the 2nd dataset - FIX LATER
    csvf = tempfile.NamedTemporaryFile(suffix='.csv')
    np.savetxt(csvf, sampling_pts.reshape(-1, sampling_pts.shape[-1]), delimiter=',', header='X,Y,Z', comments='')
    csvf.seek(0)
    coord_tab = OpenDataFile(csvf.name)
    t21 = time.time()
    coord_tab = TableToPoints(coord_tab, XColumn='X', YColumn='Y', ZColumn='Z')
    coord_tab.UpdatePipeline()
    print(f'Sampling - save/load csv: {t21 - t1} secs')
    
    #sample at coordinates
    resampled_data = ResampleWithDataset(pipeline[-1], DestinationMesh=coord_tab)
    resampled_data.UpdatePipeline()
    t22 = time.time()
    print(f'Sampling - Resample: {t22 - t21} secs')
    vtk_data = dsa.WrapDataObject(sm.Fetch(resampled_data))
    t23 = time.time()
    print(f'Sampling - Wrappers: {t23 - t22} secs')
    t2 = time.time()
    print(f'Sampling: {t2-t1} secs')

    output_arrays = []
    for field in output_fields:
        output = np.array(vtk_data.PointData[field]).reshape(*sampling_pts.shape[:-1],-1)
        output_arrays.append(output)
    output_arrays = np.concatenate(output_arrays,-1)

    pipeline = pipeline + [resampled_data, coord_tab]
    t3 = time.time()
    print(f'Reshaping: {t3-t2} secs')
    
    for pvobj in pipeline:
        Delete(pvobj)
    del pipeline, pvobj, resampled_data, coord_tab
    t4 = time.time()
    print(f'Cleanup: {t4-t3}')

    log = f'Primary pipeline: {t1-t0} secs | Sampling: {t2-t1} secs | Reshaping: {t3-t2} secs | Cleanup: {t4-t3} secs'
    print(f'{filename} log: {log}')
    
    return output_arrays

@dataclass
class PyFRVTKWriterInput:
    dtype = 'float32'
    order = None
    dataprefix = 'soln'
    precision = 'single'
    divisor = None
    gradients: bool
    outf: str
    solnf: str
    meshf: str

def process_pyfrs(pyfrm_path: str, pyfrs_path: str, output_fields: tuple[str], sampling_pts: np.ndarray, export_gradients:bool = True, filters: tuple = tuple()):

    vtuf = tempfile.NamedTemporaryFile(suffix='.vtu')
    pwi = PyFRVTKWriterInput(meshf=pyfrm_path, solnf=pyfrs_path, outf = vtuf.name, gradients = export_gradients)
    vtu_writer = PyFRVTKWriter(pwi)
    vtu_writer.write_out()

    result = sample_vtk(vtuf.name, sampling_pts, output_fields, filters=filters)
    

    return result

import multiprocessing as mp
import h5py
import time
import glob
import re

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
        -filters: tuple[paraview filter class objects, dict]. filters[k][0] is the filter to apply in the paraview data pipeline and filters[k][1] is a dict containing kwargs to pass to the initializer of filters[k][0]
        -ignore_list: tuple[str]. list of cases to ignore
        -time_window: tuple[str,str]. solutions from times outside this range will be discarded.
        -save_dtype: str. "float32" or "float64".
        '''
    def __init__(self,
                 output_path: str,
                 dataset_dir: str,
                 output_fields: tuple[str],
                 sampling_pts_sensors: np.ndarray,
                 sampling_pts_fullfield: np.ndarray,
                 filters: tuple = tuple(),
                 ignore_list: tuple[str] = tuple(),
                 time_window = (-np.inf,np.inf),
                 save_dtype = 'float32'
                 ):
        
        self.output_path = resolve_path(output_path)
        self.output_fields = output_fields
        self.filters = filters
        self.save_dtype = save_dtype

        files_list = self.get_solutions_and_meshes(resolve_path(dataset_dir), time_window=time_window)

        self.sampling_pts_sensors_n = sampling_pts_sensors.reshape(-1,sampling_pts_sensors.shape[-1]).shape[0]
        self.sampling_pts_sensors_shape = sampling_pts_sensors.shape[:-1]
        self.sampling_pts_fullfield_n = sampling_pts_fullfield.reshape(-1,sampling_pts_fullfield.shape[-1]).shape[0]
        self.sampling_pts_fullfield_shape = sampling_pts_fullfield.shape[:-1]
        self.sampling_pts = np.concatenate([sampling_pts_sensors.reshape(-1,sampling_pts_sensors.shape[-1]), sampling_pts_fullfield.reshape(-1,sampling_pts_fullfield.shape[-1])],0)

        self.read_queue = mp.Queue()
        self.write_queue = mp.Queue()

        self.soln_times_indices = {}
        for case_name, meshf, pyfrs_files, time_indices in files_list:
            self.soln_times_indices[case_name] = time_indices
            for pyfrsf in pyfrs_files:
                self.read_queue.put((case_name, meshf, pyfrsf))

    
    def _get_soln_time_from_filename(self, x, extension='.pyfrs'):
        return float(self.time_window_match.findall(x)[-1][:-len(extension)])

    def get_solutions_and_meshes(self, path, time_window = (-np.inf,np.inf)):
        solns_and_meshes = []
        for directory in filter(lambda x: os.path.isdir(f'{path}/{x}'), os.listdir(path)):
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
                
            solns_and_meshes.append((case_name, meshf, tuple(soln_files), soln_times_indices))

            
        return tuple(solns_and_meshes)
            
    def run(self):
        n_readers = mp.cpu_count()//2
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
            if self.verbose:
                rem_work_items = self.read_queue.qsize()
                print(f'Reading {self._nworkitems-rem_work_items+1}/{self._nworkitems+1}')
            
            case_name, meshf, pyfrsf = self.read_queue.get()
            soln_time = self._get_soln_time_from_filename(pyfrsf)
            tidx = self.soln_times_indices[case_name][soln_time]
            
            sampling_result = process_pyfrs(meshf, pyfrsf, self.output_fields, self.sampling_pts, filters=self.filters)
            sensor_values = sampling_result[:self.sampling_pts_sensors_n].reshape(*self.sampling_pts_sensors_shape, -1)
            full_field_values = sampling_result[self.sampling_pts_sensors_n:].reshape(*self.sampling_pts_fullfield_shape,-1)
            self.write_queue.put((case_name, tidx, sensor_values, full_field_values))
        return

    def writer_process(self):
        with h5py.File(self.output_path, 'w') as outf:
           while True:
               data = self.write_queue.get()
               if data == -1:
                   return

               case_name, tidx, sensor_values, full_field_values = data
               if case_name not in outf:
                   sensor_values_shape = [len(self.soln_times_indices[case_name]), *sensor_values.shape]
                   full_field_shape = [len(self.soln_times_indices[case_name]), *full_field_values.shape]
                   
                   sg = outf.create_group(case_name)
                   sg.create_dataset('sensors', sensor_values_shape, dtype=self.save_dtype)
                   sg.create_dataset('full_field', full_field_shape, dtype=self.save_dtype)
                   
               outf[case_name]['sensors'][tidx] = sensor_values
               outf[case_name]['full_field'][tidx] = full_field_values

               print(f'Wrote {case_name} {tidx}')

    
if __name__ == '__main__':
    c = np.stack(np.meshgrid(
        np.linspace(2.0,4.0,16),
        np.linspace(-1.0,1.0,16),
        np.linspace(2.0,4.0,16),
        indexing='ij'
    ),-1)
    nsensors = 128
    s = np.stack([5*np.ones(nsensors),0*np.ones(nsensors),np.linspace(2.0,8.0,nsensors)],-1)
    

    # z = process_pyfrs(
    #     '/storage/dataset/shape_101/shape_101.pyfrm',
    #     '/storage/dataset/shape_101/solution/soln-27.00.pyfrs',
    #     ['Pressure', 'Velocity'],
    #     c
    # )

    pm = PostprocessingManager('/tmp/test.h5', '/fr3D/pp_test',
                               output_fields=['Pressure', 'Velocity'],
                               sampling_pts_sensors=s,
                               sampling_pts_fullfield=c,
                               time_window=[20.0,60.0],
                               )
    pm.run()

    import pdb; pdb.set_trace()

    z=0

