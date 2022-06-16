import pathlib
import os
import secrets
import numpy as np
import tempfile
from dataclasses import dataclass

from paraview.simple import OpenDataFile, TableToPoints, CSVReader, ResampleWithDataset
from paraview import servermanager as sm
from paraview.vtk.numpy_interface import dataset_adapter as dsa

from pyfr.writers.vtk import VTKWriter as PyFRVTKWriter

def sample_vtk(filename: str, sampling_pts: np.ndarray, output_fields: tuple[str], filters: tuple = tuple()):

    #handle filename
    fpath = str(
        pathlib.Path(
            os.path.expanduser(filename)
        ).resolve()
    )
    

    #load file and run pipeline
    pvobj = OpenDataFile(fpath)
    pipeline = [pvobj]
    for filt in filters:
        pvobj = filt[0](pvobj, **filt[1])
        pipeline.append(pvobj)
    pipeline[-1].UpdatePipeline()

    #make a temporary csv file for the 2nd dataset - FIX LATER
    csvf = tempfile.NamedTemporaryFile(suffix='.csv')
    np.savetxt(csvf, sampling_pts.reshape(-1, sampling_pts.shape[-1]), delimiter=',', header='X,Y,Z', comments='')
    csvf.seek(0)
    coord_tab = OpenDataFile(csvf.name)
    coord_tab = TableToPoints(coord_tab, XColumn='X', YColumn='Y', ZColumn='Z')

    #sample at coordinates
    resampled_data = ResampleWithDataset(pipeline[-1], DestinationMesh=coord_tab)
    vtk_data = dsa.WrapDataObject(sm.Fetch(resampled_data))

    output_arrays = []
    for field in output_fields:
        output = np.array(vtk_data.PointData[field]).reshape(*sampling_pts.shape[:-1],-1)
        output_arrays.append(output)
    output_arrays = np.concatenate(output_arrays,-1)
    
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

    
if __name__ == '__main__':
    c = np.stack(np.meshgrid(
        np.linspace(2.0,4.0,5),
        np.linspace(-1.0,1.0,5),
        np.linspace(2.0,4.0,5),
        indexing='ij'
    ),-1)

    z = process_pyfrs(
        '/storage/dataset/shape_101/shape_101.pyfrm',
        '/storage/dataset/shape_101/solution/soln-27.00.pyfrs',
        ['Pressure', 'Velocity'],
        c
    )

    import pdb; pdb.set_trace()

