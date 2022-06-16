import vtk
import pathlib
import os
import secrets
import numpy as np
import tempfile

from paraview.simple import OpenDataFile, TableToPoints, CSVReader, ResampleWithDataset
from paraview import servermanager as sm
from paraview.vtk.numpy_interface import dataset_adapter as dsa

def sample_vtk(filename: str, sampling_pts: np.ndarray, output_field: str, filters: tuple = tuple()):

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
    output = np.array(vtk_data.PointData[output_field])
    
    return output


# c = np.stack(np.meshgrid(
#     np.linspace(2.0,4.0,50),
#     np.linspace(-1.0,1.0,50),
#     np.linspace(2.0,4.0,50),
#     indexing='ij'
# ),-1)

# o = sample_vtk('~/53.40.vtu', c, 'Grad_Velocity_U')

