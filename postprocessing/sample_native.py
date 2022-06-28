import os
import numpy as np
import h5py

from tqdm import tqdm

from pyfr.plugins.sampler import SamplerPlugin
from pyfr.readers.native import NativeReader
from pyfr.inifile import Inifile
from pyfr.mpiutil import register_finalize_handler
from pyfr.backends import get_backend
from pyfr.solvers import get_solver
from pyfr.rank_allocator import get_rank_allocation
from pyfr.mpiutil import get_comm_rank_root

class SamplerWrapper(SamplerPlugin):
    name = 'samplerwrapper'

    def _refine_pts(self, intg, elepts):
        self._element_type_order = [x.basis.name for x in intg.system.ele_map.values()]
        return super()._refine_pts(intg, elepts)

    def __call__(self, pyfrs_path):
        # MPI info
        comm, rank, root = get_comm_rank_root()

        # Get the solution matrices
        solns_file = h5py.File(pyfrs_path, 'r')
        solns = []
        for eletype in self._element_type_order:
            key = [x for x in solns_file.keys() if eletype in x][0]
            solns.append(np.array(solns_file[key]))

        # Perform the sampling and interpolation
        samples = [op @ solns[et][:, :, ei] for et, ei, _, op in self._ourpts]
        samples = self._process_samples(samples)

        # Gather to the root rank
        comm.Gatherv(samples, self._ptsrecv, root=root)

        return samples

def _convert_sampling_pts_to_pyfr_format(x):
    if isinstance(x, np.ndarray):
        x = x.reshape(-1, x.shape[-1])

    return str([tuple(z) for z in x])

def process_case(pyfrm_path, pyfrs_paths, sampling_pts, config_path=None, verbose=False):
    # Work around issues with UCX-derived MPI libraries
    os.environ['UCX_MEMTYPE_CACHE'] = 'n'
    
    # Import but do not initialise MPI
    from mpi4py import MPI

    # Manually initialise MPI
    if not MPI.Is_initialized():
        MPI.Init()

    #import pdb; pdb.set_trace()
    mesh = NativeReader(pyfrm_path)
    soln = NativeReader(pyfrs_paths[0])
    

    if config_path is not None:
        cfg = Inifile.load(config_path)
    else:
        cfg = Inifile(soln['config'])
    
    cfg.set('soln-plugin-samplerwrapper', 'samp-pts', _convert_sampling_pts_to_pyfr_format(sampling_pts))
    cfg.set('soln-plugin-samplerwrapper', 'format', 'primitive')
    cfg.set('soln-plugin-samplerwrapper', 'file', '/dev/null')
    cfg.set('soln-plugin-samplerwrapper', 'nsteps', '1')

    if hasattr(os, 'fork'):
        from pytools.prefork import enable_prefork

        enable_prefork()

    # Work around issues with UCX-derived MPI libraries
    os.environ['UCX_MEMTYPE_CACHE'] = 'n'

    # Ensure MPI is suitably cleaned up
    register_finalize_handler()

    # Create a backend
    # backend has to be not openmp for now, due to a segfault in libxsmm
    backend = get_backend('cuda', cfg)

    # Get the mapping from physical ranks to MPI ranks
    rallocs = get_rank_allocation(mesh, cfg)
    
    # Construct the solver
    solver = get_solver(backend, rallocs, mesh, soln, cfg)

    #Interpolate
    results = []
    for f in (pbar := (tqdm(pyfrs_paths) if verbose else pyfrs_paths)):
        if verbose:
            pbar.set_description(f'Processing {pyfrm_path} - {f}')
        results.append(solver.completed_step_handlers[-1](f))

    results = np.stack(results,0)

    del solver

    return results

        
if __name__ == '__main__':
    npts=16
    c = np.stack(np.meshgrid(
        np.linspace(2.0,4.0,npts),
        np.linspace(-1.0,1.0,npts),
        np.linspace(2.0,4.0,npts),
        indexing='ij'
    ),-1)
    
    pyfrm = '/fr3D/pp_test/shape_101/shape_101.pyfrm'
    pyfrs = ['/fr3D/pp_test/shape_101/solution/soln-20.00.pyfrs', '/fr3D/pp_test/shape_101/solution/soln-50.00.pyfrs']

    res = process_case(pyfrm, pyfrs, c)

    
    
