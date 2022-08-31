from mpi4py import MPI
import h5py

print("hdf5_version=" + h5py.version.hdf5_version)
print (h5py.version.hdf5_version)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    f  = h5py.File('surf_file.h5', 'r+')
    print('rank %d read and write' % rank)
else:
    f = h5py.File('surf_file.h5', 'r')
    print('rank %d read' % rank)