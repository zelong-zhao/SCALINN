import os.path,warnings
import pandas as pd
import argparse,shutil
from time import perf_counter
import numpy as np
import sys
from ML_dmft.database.dataset import AIM_Dataset
from ML_dmft.utility.mpi_tools import mpi_rank,mpi_size,mpi_comm,MPI_spread_evenly,mpi_barrier
from ML_dmft.numpy_GF.GF import complex_to_real,real_to_complex,Gtau_expand
from .G0_morebasis_triqs import expand_G0_basis_from_G0_iw

def read_args():
    parser = argparse.ArgumentParser(description='merge database')

    parser.add_argument('--merge-to-csv', action='store_true', default=False,
                help='(action) merge files from ./db/solver/1/G_tau.csv to os.path.join(db_dir, solver_searched_key.csv)')
    parser.add_argument('--reshape', action='store_true', default=False,
            help='(action) reading ./db/solver/1/G_tau.csv to ./db/solver/1/G_tau_{target_shape}.csv')
    parser.add_argument('--gen_features', action='store_true', default=False,
            help='(action) reading ./db/solver/1/{searched_key}.dat to ./db/solver/1/{searched_key}_uniform_charac_{target_shape}.dat')
    parser.add_argument('--expand_G0_basis', action='store_true', default=False,
            help='(action) expanding G0_basis G0_basis ')
    parser.add_argument('--searched_key', help='G_tau,G_l,ect.',type=str,required='--merge-to-csv' in sys.argv or '--reshape' in sys.argv)
    parser.add_argument('--target_shape', help='target_shape for G_l ect. to be G_l_target_shape',type=int,required=any(item in sys.argv for item in ['--reshape','--gen_features']))

    parser.add_argument('--solver_list', help='specified solver to process',nargs='+',default=None,type=str)
    
    args = parser.parse_args()
    return args


def merge_small_to_one(args):
    """
    FILES=./db/ED_KCL_4/0/G_tau.dat
    """

    pwd=os.getcwd()
    db_dir=os.path.join(pwd,'db')


    searched_key=args.searched_key
    print(f'only look at {searched_key}')

    if args.solver_list is not None:
        solver_list=[os.path.join(pwd,'db',_solver) for _solver in  args.solver_list]
    else:
        solver_list=find_solver(root=pwd)

    for solver in solver_list:
        dataset=AIM_Dataset(root=pwd,db='./',solver=solver,load_aim_csv_cpu=True)
        number_of_dir=len(dataset)
        Target_combined_path=os.path.join(db_dir,f"{solver}_{searched_key}.csv")

        find_del_file(Target_combined_path)
        start_time=perf_counter()
        for dat_idx in range(number_of_dir):
            G=dataset(dat_idx,searched_key)
            if searched_key in dataset.oneline_array:
                G=G.T
            elif searched_key in dataset.iw_obj:
                G=complex_to_real(G).T
            elif searched_key in dataset.single_float_key:
                G=list([G])
            writeG(Target_combined_path,G)
        print(f"time finished {perf_counter()-start_time:.5f}s")


class resize_tool():
    def __init__(self, root, db, solver):
        self.dataset=AIM_Dataset(root, db, solver)     

    def resize(self,index,key,target_shape):
        G=self.dataset(index,key)
        beta=self.dataset(index,'beta')
        if key == 'G_tau' or key == 'G0_tau':
            _,G_out=Gtau_expand(beta,target_shape,G)
        else:
            G_out=G[:target_shape]
        if index==0: 
            print(f"{key}.shape={G.shape}")
            print(f"target shape={G_out.shape}")
        return G_out
    
    def write_to_database(self,G_input,index,newkey):
        r"""
        read from ./solver/index/key.dat and write to ./solver/index/{newkey}.dat
        """
        G_input=G_input.flatten()
        target_path=self.dataset.item_path_rule(index,f"{newkey}")

        if index==0:print(f"writting to {target_path=}")
        if f"{newkey}" in self.dataset.iw_obj:
            np.savetxt(target_path,G_input,fmt='%20.10e \t %20.10e',delimiter='\t')
        elif f"{newkey}" in self.dataset.oneline_array:
            np.savetxt(target_path,G_input)
        else:
            raise ValueError(f'{newkey} not valide')
    
    def __len__(self):
        return len(self.dataset)

def database_reshape(args):
    start_time=perf_counter()

    pwd=os.getcwd()
    key=args.searched_key
    target_shape=args.target_shape

    print(f'only look at {key}')
    if args.solver_list is not None:
        solver_list=[os.path.join(pwd,'db',_solver) for _solver in  args.solver_list]
    else:
        solver_list=find_solver(root=pwd)

    for solver in solver_list:
        resize=resize_tool(pwd,'./',solver)

        if not os.path.isfile(resize.dataset.item_path_rule(0,key)):
            warnings.warn(f"{resize.dataset.item_path_rule(0,key)} not found")
            continue

        for index in range(len(resize)):
            G=resize.resize(index,key,target_shape)
            resize.write_to_database(G,index,f"{key}_{target_shape}")

    print(f"time finished {perf_counter()-start_time:.5f}s")

def find_solver(root):
    solver_list=[]
    db_dir=os.path.join(root,'db')
    for solver in os.listdir(db_dir):
        if os.path.isdir(os.path.join(db_dir,solver)): 
            print(f'find {solver}') 
            solver_list.append(solver)
    return solver_list

def find_del_file(path):
    if os.path.isfile(path):
        os.remove(path)
        print(f'{"delete":10.5} {path}')
    print(f'{"writting to":10.5} {path}')

def accessG(G_path):
    G=[]
    with open(G_path,'r') as F:
        for lines in F:
            G.append(float(lines))
    G=np.array(G).reshape((1,len(G)))
    return G

def writeG(G_path,G):
    with open(G_path,'a') as F:
        np.savetxt(F,G)


def create_features(args):
    r"""
     reading ./db/solver/1/{searched_key}.dat 
     
     to ./db/solver/1/{searched_key}_uniform_charac_{target_shape}.dat
    """
    from ML_dmft.torch_model.transformer_charac.linear_character import proceed_chrac

    start_time=perf_counter()
    character_name='uniform'

    pwd=os.getcwd()
    root=pwd
    db='./'
    key=args.searched_key
    target_shape=args.target_shape

    print(f'only look at {key}')
    print(f'going to produce {key}_{character_name}_{target_shape}')
    newkey=f'{key}_{character_name}_{target_shape}'
    if args.solver_list is not None:
        solver_list=[os.path.join(pwd,'db',_solver) for _solver in  args.solver_list]
    else:
        solver_list=find_solver(root=pwd)

    warnings.warn('args.searched_key. G(tau) and G(l) only')
    for solver in solver_list:
        print(f"{solver=}")
        dataset=AIM_Dataset(root, db, solver)

        if not os.path.isfile(dataset.item_path_rule(0,key)):
            warnings.warn(f"{dataset.item_path_rule(0,key)} not found")
            continue

        for index in range(len(dataset)):
            target_path=dataset.item_path_rule(index,f"{newkey}")
            g=dataset(index,key)
            feature=proceed_chrac(g,32)
            np.savetxt(target_path,feature)
        
    print(f"{(perf_counter()-start_time):.5f}s")

def expand_G0_basis(args):
    if mpi_rank()==0:
        start_time=perf_counter()
        if mpi_size()==1: warnings.warn('this module is mpi-enabled')

    pwd=os.getcwd()
    root=pwd
    db='./'

    if mpi_rank()==0:
        solver_list=find_solver(root=pwd)
    else:
        solver_list=None

    mpi_barrier()
    solver_list=mpi_comm().bcast(solver_list, root=0)
    
    for solver in solver_list:

        dataset=AIM_Dataset(root, db, solver)
        if mpi_rank() == 0: print(f"{solver=}")
            
        if not os.path.isfile(dataset.item_path_rule(0,'G0_iw')):
            if mpi_rank() == 0: warnings.warn(f"{dataset.item_path_rule(0,'G0_iw')} not found")
            continue

        index_list,params_list=MPI_spread_evenly(input_list=np.arange(len(dataset)),
                                            whoami=mpi_rank(),
                                            size=mpi_size())
        mpi_barrier()
        for index in index_list:

            g0_tau,g0_l=expand_G0_basis_from_G0_iw(root,db,solver,index)

            g0_tau_name=dataset.item_path_rule(index,'G0_tau')
            g0_l_name=dataset.item_path_rule(index,'G0_l')

            if index==0:
                print(f'saving to {g0_tau_name=} {g0_tau.shape=}')
                print(f'saving to {g0_l_name=} {g0_l.shape=}')

            np.savetxt(g0_tau_name,g0_tau)
            np.savetxt(g0_l_name,g0_l)

        mpi_barrier()

    if mpi_rank()==0:
        print(f"time to finish is {(perf_counter()-start_time):.5f}")


def main():
    args=read_args()
    
    if args.merge_to_csv:
        merge_small_to_one(args)

    if args.reshape:
        database_reshape(args)

    if args.gen_features:
        create_features(args)

    if args.expand_G0_basis:
        expand_G0_basis(args)