from time import perf_counter
import os
from ML_dmft.solvers.ED_KCL_solver import RUN_ED
from ML_dmft.utility.tools import read_params,save2pkl,save2csv,dump_to_yaml
from ML_dmft.utility.mpi_tools import mpi_rank,mpi_size,mpi_gather,mpi_barrier,MPI_spread_evenly,MPI_Collect_evenly

def solve_AIM(args):    
    """
    Main routine for solving AIM for the database
    """
    
    solver_params={}
    solver_params['gf_param']=args['gf_param']
    
    solver_params['solve dmft']=args['dmft_param']

    for key in args['ed_params']:
        solver_params['solve dmft'][key]=args['ed_params'][key]

    ED_bath_site=args['ed_params']['ED_min_bath']

    aim_params_file = args['main_params']['aim_params_file']
    idx2cal = args['main_params']['item_idx']
    out_solver_name =  args['main_params']['out_solver_name']
    if out_solver_name is None:
        out_solver_name = f"ED_KCL_{str(ED_bath_site)}"

    params = read_params(aim_params_file)

    verbose=args['gf_param']['verbose']

    ##example input
    if mpi_rank() == 0 and verbose >= 2:
        print(solver_params)
        dump_to_yaml('out_ED_solver_params.yml',solver_params)
        dump_to_yaml('out_sample_aim_params.yml',params[0])

    whoami=mpi_rank()
    _mpi_size_=mpi_size()

    if whoami==0:
        time_begin=perf_counter()
        print("*"*20, " Running ED_KCL", "*"*20)

    index_list,params_list=MPI_spread_evenly(input_list=params,
                                            whoami=whoami,
                                            size=_mpi_size_)
    
    for idx,param in enumerate(params_list):
        if idx2cal is not None: 
            idx,param = idx2cal,params_list[idx2cal]
            print(f"Calculating {idx=} {param=}")
        
        ED=RUN_ED(param,solver_params)

        ED.save_as_small_files(param=param,root=os.path.join('./db',out_solver_name,str(index_list[idx])))
        if verbose >= 2: print(f"Saving to: {os.path.join('./db',out_solver_name,str(index_list[idx]))}")

        if idx2cal is not None: 
            break

    if whoami==0:       
        if idx%10 == 0:
            print('total number of points: {}, calculating{}'.format(len(params),idx))

    mpi_barrier()

    if whoami==0:
        time_finish=perf_counter()-time_begin

        print('*'*20,time_finish,'*'*20)