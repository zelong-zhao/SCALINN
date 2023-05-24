from ML_dmft.solvers.PT_solver import PT_solver
from typing import Tuple 
import typing

import numpy

def pt_interface(args:dict,U:float,g_iw:numpy.ndarray,W:float=1)-> typing.Tuple[numpy.ndarray,float]:
    solver_params={}
    solver_params['gf_param']=args['gf_param']

    solver_params['solve dmft']=args['dmft_param']
    for key in args['ipt_param']:
        solver_params['solve dmft'][key]=args['ipt_param'][key]

    IPT = PT_solver(solver_params['gf_param'])
    param = dict(U=U,eps=-U/2.,W=W)
    IPT.Bethe_DMFT_Giw_inputs(param=param,dmft_param=solver_params['solve dmft'],g_iw=g_iw)
    g_iw_out,z=IPT.get_output_giw_z()
    return g_iw_out,z