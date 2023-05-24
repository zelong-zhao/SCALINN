from mpi4py import MPI
import os
import numpy as np 
import itertools

# parallel stuff
def mpi_rank():
    comm = MPI.COMM_WORLD
    return comm.Get_rank()

def mpi_size():
    comm = MPI.COMM_WORLD
    return comm.Get_size()

def mpi_comm():
    comm = MPI.COMM_WORLD
    return comm

def mpi_barrier():
    comm = MPI.COMM_WORLD
    return comm.Barrier()

def mpi_gather(data,root):
    return MPI.COMM_WORLD.gather(data,root=root)


def parallel_info():
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())
    print(mpi_rank(),'of',mpi_size())


def MPI_spread_evenly(input_list,whoami,size):
    """
    Input list split by mpi rank
    whoami: rank
    """
    index_list,item_list=[],[]
    for idx,item in enumerate(input_list):
        if idx%size!=whoami:continue
        index_list.append(idx)
        item_list.append(item)
    return index_list,item_list

def MPI_Collect_evenly(index_list:list,item_list:list)-> list:
    """
    proceed MPI.COMM_WORLD.gather
    """
    item_list=np.array(list(itertools.chain(*item_list)))
    index_list=np.array(list(itertools.chain(*index_list)))
    item_list = item_list[index_list.argsort()].tolist()
    return item_list

def MPI_Collect_evenly_file(index_list,file_list):
    """
    input
    -----
    file_list,index_list

    file_list[rank] is the path of a file.

    index_list[index] return origin index of the array

    important
    ------
    data in file_list should be one row for get tracking by index_list
    """
    index_list=np.array(list(itertools.chain(*index_list)))

    data_read=[]
    for file in file_list:
        data=np.genfromtxt(file)
        data_read.append(data)
    data_read=np.array(data_read)
    data_read=np.row_stack(data_read)
    data_read=data_read[index_list.argsort()].tolist()
    return data_read

def main():
    size=mpi_size()
    whoami=mpi_rank()
    input_list=np.arange(8,dtype=int)

    index_list,item_list=MPI_spread_evenly(input_list=input_list,
                                            whoami=whoami,
                                            size=size)

    mpi_barrier()

    item_list=MPI.COMM_WORLD.gather(item_list,root=0)
    index_list=MPI.COMM_WORLD.gather(index_list,root=0)

    if whoami==0:

        """
        item_list are [[result from rank0],[...rank1],...]
        """

        item_list=np.array(list(itertools.chain(*item_list)))
        index_list=np.array(list(itertools.chain(*index_list)))

        """
        verified, index_list unstack one level of stacked list
        """

        item_list = item_list[index_list.argsort()].tolist()

        print('item list',item_list)

if __name__ == '__main__':
    main()
