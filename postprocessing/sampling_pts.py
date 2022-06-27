import numpy as np
import h5py
import pydscpack
import itertools
from collections.abc import Iterable

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

class GridSamplingPts(BaseSamplingPts):
    name= 'grid'
    __doc__ = 'Sampling pts on a grid'
    def __init__(self, linspace_args):
        self._pts = np.stack(np.meshgrid(*[np.linspace(*l) for l in linspace_args], indexing='ij'), -1)

        super().__init__()

def _vertex_renumbering_filter_nodemap(x):
    out = {}
    min_encountered = {}
    for ele_type, npts in x:
        if (ele_type not in out) or (min_encountered[ele_type] > npts):
            out[ele_type] = x[(ele_type, npts)]
            min_encountered[ele_type] = npts

    return out
        
class PyfrmSamplingPts(BaseSamplingPts):
    #yani yukaridaki ornege bakacak olursak, 188 nolu elemanin 2 nolu yuzu BC uzerinde. mapping'e bakiyorum, _petype['hex'][2] = [1,2,5,6]. 1,2,5,6'yi yukarida koydugun array'e gore 1,3,5,7 yapiyorum sonra spt_hex_p0'a bakiyorum. kisacasi mesh['spt_hex_p0'][<1,3,5 ve 7>, 188,:] bana vertex koordinatlarini veriyor.
    name='pyfrm-base'
    _petype_fnmap = GmshReader._petype_fnmap
    _vertex_renumbering = _vertex_renumbering_filter_nodemap(GmshReader._nodemaps)
    __doc__ = 'Sampling pts generator for defining sampling points relative to physical groups in pyfrm files'

    @classmethod
    def _extract_physical_grp_vertices(cls, pyfrm, physgrp, extrusion_dim=2):
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

        vertices = np.array(tuple(vertices))

        #if the object is an object extruded along a spatial dim, return coords of its
        #2 dimensional cross section and the extents in the extruded direction.
        #else, just return the vertices.
        if extrusion_dim is not None:
            extr_dim_coords = np.sort(np.unique(vertices[:,extrusion_dim]))
            cross_section_coords = np.array(list(
                map(lambda x: [x[i] for i in filter(lambda c: c != extrusion_dim, range(x.shape[0]))],
                    filter(lambda x: x[extrusion_dim]==extr_dim_coords[0], vertices)
                    )))
            return cross_section_coords, extr_dim_coords
        else:        
            return vertices

    @classmethod
    def _extract_domain_extrema(cls, pyfrm):
        with h5py.File(pyfrm, 'r') as mesh:
            per_ele_type_mins = []
            per_ele_type_maxes = []

            for ele_field in filter(lambda x: x[:3] == 'spt', mesh):
                vertex_coords = np.array(mesh[ele_field]).reshape(-1, mesh[ele_field].shape[-1])
                mins = np.min(vertex_coords, axis=0)
                per_ele_type_mins.append(mins)
                maxes = np.max(vertex_coords, axis=0)
                per_ele_type_maxes.append(maxes)

        mins = np.min(np.stack(per_ele_type_mins,0), axis=0)
        maxes = np.max(np.stack(per_ele_type_maxes,0), axis=0)

        return mins, maxes
                
class PyfrmRelativeSamplingPts(PyfrmSamplingPts):
    name='pyfrm-relative'
    __doc__='Places points supplied in ptsreader relative to the centroid, minimum or maximum of the specified physical group along each dimension'
    def __init__(self, pyfrm, physgrp, ptsreader, rel_to=('centroid', 'centroid', 'centroid')):

        if isinstance(ptsreader, dict):
            ptsreader = get_samplingptshandler(**ptsreader)
        elif isinstance(ptsreader, Iterable):
            ptsreader = LiteralSamplingPts(ptsreader)
        else:
            assert isinstance(ptsreader, BaseSamplingPts)
        read_pts = ptsreader.pts

        vertices = self._extract_physical_grp_vertices(pyfrm, physgrp, extrusion_dim=None)
        pts_of_interest = {
            'centroid': np.mean(vertices, axis=0),
            'minimum': np.min(vertices, axis=0),
            'maximum': np.max(vertices, axis=0)
        }

        for axis,mode in enumerate(rel_to):
            read_pts[:,axis] += pts_of_interest[mode][axis]

        self._pts = read_pts

        super().__init__()
        
class PyfrmAnnulusGridSamplingPts(PyfrmSamplingPts):
    name='pyfrm-annulus'
    __doc__='''
    Takes the specified physical group (physgrp) in the pyfrm file, calculates the outer boundaries of the fluid domain and computes a Schwarz-Christoffel mapping the fluid domain to an (extruded) annulus. Samples a regular grid, equispaced in the r, theta and z directions, on the annular cylinder and maps the points back to the original fluid domain.
    '''
    
    def __init__(self, pyfrm, physgrp, npts, extrusion_dim=2, outer_box=None):
        cross_sxn, extr_dim_coords  = self._extract_physical_grp_vertices(pyfrm, physgrp, extrusion_dim=extrusion_dim)
        extr_dim_min, extr_dim_max = np.min(extr_dim_coords), np.max(extr_dim_coords)
        # extr_dim_coords = np.ones([npts[extrusion_dim], 3], dtype=extr_dim_coords)
        # extr_dim_coords[:,extrusion_dim] = np.linspace(np.min(extr_dim_coords), np.max(extr_dim_coords), npts[extrusion_dim])

        if outer_box is None:
            domain_mins, domain_maxes = self._extract_domain_extrema(pyfrm)
            domain_mins = np.delete(domain_mins, extrusion_dim)
            domain_maxes = np.delete(domain_maxes, extrusion_dim)
            domain_extents = [(start, end) for start, end in zip(domain_mins, domain_maxes)]
            outer_box = np.array([(domain_extents[0][0],domain_extents[1][0]),(domain_extents[0][0],domain_extents[1][1]),(domain_extents[0][1],domain_extents[1][0]), (domain_extents[0][1],domain_extents[1][1])])
        elif isinstance(outer_box, dict):
            outer_box_sampler = get_samplingptshandler(**outer_box)
            outer_box = outer_box_sampler.pts
        else:
            outer_box = np.array(outer_box)
            assert len(outer_box.shape) >= 2 and outer_box.shape[-1] == 3
        
        cross_sxn_complex = cross_sxn[:,0]+1j*cross_sxn[:,1]
        outer_box_complex = outer_box[:,0]+1j*outer_box[:,1]

        amap = pydscpack.AnnulusMap(outer_box_complex, cross_sxn_complex)
        annulus_coords_complex, phys_coords_complex = amap._generate_annular_grid(n_pts=np.delete(npts,extrusion_dim))
        phys_coords_indices = np.delete([0,1,2], extrusion_dim)
        phys_coords = np.ones([*phys_coords_complex.shape,3])
        phys_coords[...,phys_coords_indices[0]] = np.real(phys_coords_complex)
        phys_coords[...,phys_coords_indices[1]] = np.imag(phys_coords_complex)
        phys_coords = np.expand_dims(phys_coords, extrusion_dim)

        extr_dim_coords_shape = [1 for _ in phys_coords.shape]
        extr_dim_coords_shape[extrusion_dim] = npts[extrusion_dim]
        extr_dim_coords_shape[-1] = phys_coords.shape[-1]
        extr_dim_coords = np.ones(extr_dim_coords_shape, dtype=phys_coords.dtype)
        extr_dim_coords[...,extrusion_dim] = np.linspace(extr_dim_min, extr_dim_max, npts[extrusion_dim])

        grid_coords = phys_coords * extr_dim_coords

        self._pts = grid_coords

        super().__init__()

        
        

class PyfrmCartesianGridSamplingPts(PyfrmSamplingPts):
    name='pyfrm-cartesian'
    def __init__(self, pyfrm, npts):
        mins, maxes = self._extract_domain_extrema(pyfrm)

        self._pts = np.stack(np.meshgrid(*[np.linspace(m,M,n) for m,M,n in zip(mins, maxes, npts)], indexing='ij'),-1)
        super().__init__()


def get_all_subclasses(cls):
    subclass_list = []

    def recurse(klass):
        for subclass in klass.__subclasses__():
            subclass_list.append(subclass)
            recurse(subclass)

    recurse(cls)

    return set(subclass_list)


def get_samplingptshandler(name, **kwargs):

    try:
        samplingpts_class = next(filter(lambda x: x.name == name, get_all_subclasses(BaseSamplingPts)))
    except:
        opts = [x.name for x in get_all_subclasses(BaseSamplingPts)]
        raise(RuntimeError(f'Could not find sampling pts handler {name} - available options {opts}'))

    sampler = samplingpts_class(**kwargs)

    return sampler

if __name__ == '__main__':
    test_pyfrm = '/fr3D/pp_test2/shape_2091/shape_2091.pyfrm'
    # z = PyfrmAnnulusGridSamplingPts(test_pyfrm, 'obstacle', [64,256,64])
    # import pdb; pdb.set_trace()
    # zz = 1
    # print(PyfrmSamplingPts._petype_fnmap)
    # print(PyfrmSamplingPts._vertex_renumbering)
    # vert = PyfrmSamplingPts._extract_physical_grp_vertices(test_pyfrm, 'obstacle')
    # extr = PyfrmSamplingPts._extract_domain_extrema(test_pyfrm)
    # import pdb; pdb.set_trace()
    # z=1
        
