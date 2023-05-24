import triqs.gf
import os
import numpy as np
from .dataset import AIM_Dataset_Rich
from ML_dmft.triqs_interface.v2 import triqs_interface
from ML_dmft.solvers.ED_KCL_solver import get_data_basis


def expand_G0_basis_from_G0_iw(root,db,solver,index):

    dataset=AIM_Dataset_Rich(root=root,
                        db=db,
                        solver=solver,
                        index=index)
    target_n_tau=dataset.n_tau
    
    if index==0:
        print(f'{target_n_tau=}')

    triqs_obj=triqs_interface(dataset)
    triqs_obj.G0_more_basis()
    g0_tau,g0_l=extract_data_G0(triqs_obj.G0_tau,triqs_obj.G0_l,target_n_tau)
    return g0_tau.T,g0_l.T


def extract_data_G0(G0_tau,G0_l,target_n_tau):

    data_tau = np.asarray(get_data_basis(G0_tau, "tau"))

    idxs = np.linspace(0, len(data_tau[0]) - 1,target_n_tau).astype(int)
    new_tau=np.zeros((1,len(idxs)))
    for i,idx in enumerate(idxs):
        new_tau[0][i]=data_tau[1][idx]

    data_leg = np.asarray(get_data_basis(G0_l, "legendre"))
    ret_leg = data_leg[1]
    ret_leg = ret_leg.reshape((1,len(ret_leg)))
    return new_tau, ret_leg