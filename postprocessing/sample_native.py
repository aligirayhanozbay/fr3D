import os
import re
import tempfile
import shutil
import numpy as np

from collections import defaultdict
from dataclasses import dataclass

from pyfr.plugins.sampler import SamplerPlugin
from pyfr.readers.native import NativeReader
from pyfr.inifile import Inifile
from pyfr.mpiutil import register_finalize_handler
from pyfr.backends import get_backend
from pyfr.solvers import get_solver
from pyfr.rank_allocator import get_rank_allocation
from pyfr.partitioners import BasePartitioner, get_partitioner
from pyfr.util import subclasses
from pyfr.writers.native import NativeWriter, write_pyfrms
from pyfr.mpiutil import get_comm_rank_root

class SamplerWrapper(SamplerPlugin):
    name = 'samplerwrapper'

    def __call__(self, intg):
        # MPI info
        comm, rank, root = get_comm_rank_root()

        # Get the solution matrices
        solns = intg.soln

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

def load_pyfr_integrator(pyfrm_path, pyfrs_path, sampling_pts, config_path=None):
    # Work around issues with UCX-derived MPI libraries
    os.environ['UCX_MEMTYPE_CACHE'] = 'n'
    
    # Import but do not initialise MPI
    from mpi4py import MPI

    # Manually initialise MPI
    if not MPI.Is_initialized():
        MPI.Init()

    #import pdb; pdb.set_trace()
    mesh = NativeReader(pyfrm_path)
    soln = NativeReader(pyfrs_path)
    

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
    backend = get_backend('openmp', cfg)

    # Get the mapping from physical ranks to MPI ranks
    rallocs = get_rank_allocation(mesh, cfg)
    
    # Construct the solver
    solver = get_solver(backend, rallocs, mesh, soln, cfg)

    #do something like
    #mp = solver.completed_step_handlers[-1](solver)
    #find a way to update solver.soln to sample data from multiple files at once

    return solver
    
        
if __name__ == '__main__':
    c = np.stack(np.meshgrid(
        np.linspace(2.0,4.0,64),
        np.linspace(-1.0,1.0,64),
        np.linspace(2.0,4.0,64),
        indexing='ij'
    ),-1)
    
    pyfrm = '/fr3D/pp_test/shape_101/shape_101.pyfrm'
    pyfrs = '/fr3D/pp_test/shape_101/solution/soln-50.00.pyfrs'

    intg = load_pyfr_integrator(pyfrm, pyfrs, c)
