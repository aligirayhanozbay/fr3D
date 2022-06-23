import numpy as np
import h5py

from pyfr.readers.gmsh import GmshReader

class BaseSamplingPts:
    name = 'base'
    def __init__(self):
        assert self._pts.shape[-1] == 3
    
    @property
    def pts(self):
        return self._pts

class LiteralSamplingPts(BaseSamplingPts):
    name = 'literal'
    def __init__(self, pts):
        self._pts = np.array(pts, dtype=np.float64)
        
        super().__init__()
    
class TxtSamplingPts(BaseSamplingPts):
    name = 'csv'
    __doc__ = 'Sampling pts reader for csv files'
    def __init__(self, f, delimiter=','):
        self._pts = np.genfromtxt(f, delimiter=delimiter, dtype=np.float64)

        super().__init__()

class NumpySamplingPts(BaseSamplingPts):
    name = 'numpy'
    __doc__ = 'Sampling pts reader for npy files'
    def __init__(self, f):
        self._pts = np.load(f).astype(np.float64)

        super().__init__()

def _vertex_renumbering_filter_nodemap(x):
    out = {}
    min_encountered = {}
    for ele_type, npts in x:
        if (ele_type not in out) or (min_encountered[ele_type] > npts):
            out[ele_type] = x[(ele_type, npts)]
            min_encountered[ele_type] = npts

    return out
        
class PyfrmGroupRelativeSamplingPts(BaseSamplingPts):
    #yani yukaridaki ornege bakacak olursak, 188 nolu elemanin 2 nolu yuzu BC uzerinde. mapping'e bakiyorum, _petype['hex'][2] = [1,2,5,6]. 1,2,5,6'yi yukarida koydugun array'e gore 1,3,5,7 yapiyorum sonra spt_hex_p0'a bakiyorum. kisacasi mesh['spt_hex_p0'][<1,3,5 ve 7>, 188,:] bana vertex koordinatlarini veriyor.
    name='pyfrm'
    _petype_fnmap = GmshReader._petype_fnmap
    _vertex_renumbering = _vertex_renumbering_filter_nodemap(GmshReader._nodemaps)
    __doc__ = 'Sampling pts generator for defining sampling points relative to physical groups in pyfrm files'
    
    def __init__(self, pyfrm, physgrp):
        pass

    @classmethod
    def _read_pyfrm(cls, pyfrm, physgrp):
        ele_vertices_on_physgrp = []
        with h5py.File(pyfrm, 'r') as mesh:
            faces = np.concatenate([mesh[v] for v in filter(lambda x: f'bcon_{physgrp}_p' in x, mesh)],0)
            vertices = set()
            for face in faces:
                ele_type, ele_num, face_idx, partition = face
                ele_type = ele_type.decode('ascii')
                ele_field = f'spt_{ele_type}_p{partition}'

                #TODO: fix hardcoded quad face type
                gmsh_vertex_indices = cls._petype_fnmap[ele_type]['quad'][face_idx]
                pyfr_vertex_indices = [cls._vertex_renumbering[ele_type][v] for v in gmsh_vertex_indices]

                coords = [mesh[ele_field][v,ele_num,:] for v in pyfr_vertex_indices]
                for coord in coords:
                    coord_tuple = tuple(coord)
                    vertices.add(coord_tuple)
                    
        return vertices
                
    

def get_samplingptshandler(name, **kwargs):

    try:
        samplingpts_class = next(filter(lambda x: x.name == name, BaseSamplingPts.__subclasses__()))
    except:
        opts = [x.name for x in BaseSamplingPts.__subclasses__()]
        raise(RuntimeError(f'Could not find sampling pts handler {name} - available options {opts}'))

    sampler = samplingpts_class(**kwargs)

    return sampler

if __name__ == '__main__':
    test_pyfrm = '/fr3D/pp_test2/shape_2091/shape_2091.pyfrm'
    print(PyfrmGroupRelativeSamplingPts._petype_fnmap)
    print(PyfrmGroupRelativeSamplingPts._vertex_renumbering)
    res = PyfrmGroupRelativeSamplingPts._read_pyfrm(test_pyfrm, 'obstacle')
    import pdb; pdb.set_trace()
    z=1
        
