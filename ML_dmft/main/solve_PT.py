from time import perf_counter
from ML_dmft.solvers.PT_solver import RUN_IPT
from ML_dmft.utility.db_gen import read_params
from ML_dmft.utility.tools import save2pkl,save2csv,dump_to_yaml
from ML_dmft.utility.mpi_tools import mpi_rank,mpi_size,mpi_gather,mpi_barrier,MPI_spread_evenly,MPI_Collect_evenly

def solve_AIM(args):
    """
    Main routine for solving AIM for the database
    """

    solver_params={}
    solver_params['gf_param']=args['gf_param']

    solver_params['solve dmft']=args['dmft_param']
    for key in args['ipt_param']:
        solver_params['solve dmft'][key]=args['ipt_param'][key]

    save_csv=args['output_format']['save_csv']
    save_pkl_dict=args['output_format']['save_pkl_dict']
    save_small_files=args['output_format']['save_small_files']


    params = read_params("./db/aim_params.csv")

    verbose=args['gf_param']['verbose']
    ##example input
    if mpi_rank() == 0 and verbose >= 2:
        print(solver_params)
        dump_to_yaml('out_PT_solver_params.yml',solver_params)
        dump_to_yaml('out_sample_aim_params.yml',params[0])
    
    whoami=mpi_rank()
    _mpi_size_=mpi_size()

    if whoami==0:
        time_begin=perf_counter()
        print("*"*20, " Running IPT ", "*"*20)

    index_list,params_list=MPI_spread_evenly(input_list=params,
                                            whoami=whoami,
                                            size=_mpi_size_)

    if save_pkl_dict: 
        data_list = []
    if save_csv: 
        G_tau_list,G_leg_list = [],[]
        G0_tau_list,G0_leg_list = [],[]

    for idx,param in enumerate(params_list):
        IPT=RUN_IPT(param,solver_params)
        if save_small_files:
            IPT.save_as_small_files(param=param,root='./db/IPT/{}'.format(int(index_list[idx])))
        if save_pkl_dict: 
            data_list.append(IPT.saveGF2mydataformat(param))
        if save_csv:
            G_tau,G_leg=IPT.extract_data()
            G_tau_list.append(G_tau)
            G_leg_list.append(G_leg)

            G0_tau,G0_leg=IPT.extract_data_G0()
            G0_tau_list.append(G0_tau)
            G0_leg_list.append(G0_leg)

    if whoami==0:       
        if idx%10 == 0:
            print('total number of points: {}, calculating{}'.format(len(params),idx))

    mpi_barrier()
    index_list=mpi_gather(index_list,root=0)
    if save_pkl_dict: 
        data_list=mpi_gather(data_list,root=0)
    if save_csv:
        G_tau_list=mpi_gather(G_tau_list,root=0)
        G_leg_list=mpi_gather(G_leg_list,root=0)

        G0_tau_list=mpi_gather(G0_tau_list,root=0)
        G0_leg_list=mpi_gather(G0_leg_list,root=0)

    if whoami==0:
        time_finish=perf_counter()-time_begin

        if save_pkl_dict:
            data_list=MPI_Collect_evenly(index_list,data_list)
            save2pkl(input_list=data_list,fname='./db/IPT.pkl')
            print(len(data_list))

        if save_csv:
            G_tau_list=MPI_Collect_evenly(index_list,G_tau_list)
            G_leg_list=MPI_Collect_evenly(index_list,G_leg_list)
            
            G0_tau_list=MPI_Collect_evenly(index_list,G0_tau_list)
            G0_leg_list=MPI_Collect_evenly(index_list,G0_leg_list)
            
            
            print(f'number of files: {len(G_tau_list)}, {len(G_leg_list)}')


            save2csv(G_tau_list,fname='./db/IPT_G_tau.csv')
            save2csv(G_leg_list,fname='./db/IPT_G_leg.csv')

            save2csv(G0_tau_list,fname='./db/IPT_G0_tau.csv')
            save2csv(G0_leg_list,fname='./db/IPT_G0_leg.csv')

        print('*'*20,time_finish,'*'*20)
