from time import perf_counter
from ML_dmft.solvers.PT_solver import RUN_IPT
from ML_dmft.solvers.HI_solver import RUN_HubbardI
from ML_dmft.utility.db_gen import read_params
from ML_dmft.utility.tools import save2pkl,save2csv,dump_to_yaml
from ML_dmft.utility.mpi_tools import mpi_rank,mpi_size,mpi_gather,mpi_barrier,MPI_spread_evenly,MPI_Collect_evenly

def solve_AIM(args:dict,solver2use:str='HubbardI'):
    """
    Main routine for solving AIM for the database
    """
    solver_params={}
    solver_params['gf_param']=args['gf_param']

    solver_params['solve dmft']=args['dmft_param']

    params = read_params("./db/aim_params.csv")
    verbose=args['gf_param']['verbose']

    idx2cal = args['main_params']['item_idx']

    whoami=mpi_rank()
    _mpi_size_=mpi_size()

    if solver2use == 'HubbardI':
        IMP_SOLVER=RUN_HubbardI   
        out_solver_name = 'HubbardI'     

    elif solver2use == 'IPT':
        IMP_SOLVER=RUN_IPT
        out_solver_name = 'IPT'
        for key in args['ipt_param']:
            solver_params['solve dmft'][key]=args['ipt_param'][key]

    else:
        raise TypeError('no such solver')

    ##example input
    if mpi_rank() == 0 and verbose >= 2:
        print(solver_params)
        dump_to_yaml(f'out_{out_solver_name}_params.yml',solver_params)
        dump_to_yaml('out_sample_aim_params.yml',params[0])    


    if whoami==0:
        time_begin=perf_counter()
        print("*"*20, f" Running {out_solver_name} ", "*"*20)

    index_list,params_list=MPI_spread_evenly(input_list=params,
                                            whoami=whoami,
                                            size=_mpi_size_)

    for idx,param in enumerate(params_list):
        if idx2cal is not None: 
            idx,param = idx2cal,params_list[idx2cal]
        Solved=IMP_SOLVER(param,solver_params)
        Solved.save_as_small_files(param=param,root=f'./db/{out_solver_name}/{int(index_list[idx])}')
        if idx2cal is not None: 
            break

    if whoami==0:       
        if idx%10 == 0:
            print('total number of points: {}, calculating{}'.format(len(params),idx))

    mpi_barrier()
    # index_list=mpi_gather(index_list,root=0)

    if whoami==0:
        time_finish=perf_counter()-time_begin

        print('*'*20,time_finish,'*'*20)
