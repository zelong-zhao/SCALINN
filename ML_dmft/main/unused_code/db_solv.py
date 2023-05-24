import numpy as np
import random
import time
import os.path
from subprocess import Popen
import pandas as pd
from triqs.plot.mpl_interface import oplot, plt
from ML_dmft.solvers.PT_solver import *
from ML_dmft.solvers.ED_KCL_solver import *
from ML_dmft.solvers.solv_funcs import * 
from ML_dmft.utility.db_gen import *
from ML_dmft.utility.tools import *
from ML_dmft.utility.read_config import *

def solve_AIM(system_param, db_param, aim_param):
    """
    Main routine for solving AIM for the database
    """
    time_begin=time.time()

    beta = system_param["beta"]
    U_ = db_param["U_"]
    eps_ = db_param["eps_"]
    D_ = db_param["D_"]
    V_ = db_param["V_"]
    N_ = db_param["N_"]
    database_distribution = db_param["database_distribution"]
    plot_hybrid_ = db_param["database_plot_hybrid"]
    samples_per_file = system_param["n_samples"]
    basis = system_param["basis"]
    n_l = db_param["n_l"]
    poly_semilog = db_param["poly_semilog"]
    n_iw = db_param["n_iw"]
    n_tau = db_param["n_tau"]
    target_n_tau = system_param["tau_file"]-1 # ES BUG: shouldn't have to -1 here, but crash if not
    writing = aim_param["aim_writing"]
    solvers = aim_param["approximations"]
    
    indices = [0] # ES TODO: should it be hardcoded?
    
    whoami = mpi_rank()
    solv_param = {"beta": beta,
                  "n_l": n_l,
                  "n_iw": n_iw,
                  "n_tau": n_tau,
                  "indices": indices,
                  "basis": basis,
                  "target_n_tau": target_n_tau,
                  "writing": writing}
   

    plot_param = {"sample_max": 4,
                  "chosen_rank": 0,
                  "basis": basis,
                  "poly_semilog": poly_semilog}
    
    params = read_params_MPI("./db/aim_params.csv", whoami, samples_per_file)
    
    if solvers["IPT"]:
        if whoami == 0:
            time_ipt_begin=time.time()
            print("*"*20, " Running IPT ", "*"*20)
            print("")
        delete_files("G_IPT", beta)        
        ipt_tau_ = []
        ipt_leg_ = []
        ipt_iw_ = []
        for p in params:            
            IPT = PT_solver(solv_param)
            IPT.Delta_iw << hyb(p["E_p"], p["V_p"])
            IPT.G_0_iw << inverse(iOmega_n - p["eps"] - IPT.Delta_iw)
            IPT.solve(param=p)
            
            ipt_tau, ipt_leg = IPT.extract_data()
            ipt_iw = IPT.extract_data_iw()
            ipt_tau_.append(ipt_tau)
            ipt_leg_.append(ipt_leg)
            ipt_iw_.append(ipt_iw)
        mpi_barrier()
        ipt_leg_ = MPI.COMM_WORLD.gather(ipt_leg_, root=0)
        ipt_tau_ = MPI.COMM_WORLD.gather(ipt_tau_, root=0)
        ipt_iw_ = MPI.COMM_WORLD.gather(ipt_iw_, root=0)
        if whoami == 0:
            time_ipt_finished=time.time()-time_ipt_begin
            print('*'*20,time_ipt_finished,'*'*20)
            fname="./db/G_IPT_leg.csv"
            write_me_MPI_GFs(fname, ipt_leg_)
            fname="./db/G_IPT_tau.csv"
            write_me_MPI_GFs(fname, ipt_tau_)
            fname="./db/G_IPT_iw.csv"
            write_me_MPI_GFs(fname, ipt_iw_)
        mpi_barrier()

    # if solvers["HI"]: 
    #     if whoami == 0:
    #         print("*"*20, " Running HI ", "*"*20)
    #         print("")
    #     delete_files("G_HI", beta)
    #     HI_tau_ = []
    #     HI_leg_ = []
    #     for p in params:            
    #         HI = HI_solver(solv_param)
    #         HI.Delta_iw << hyb(p["E_p"], p["V_p"])
    #         HI.solve(param=p)
    #         HI_tau, HI_leg = HI.extract_data()
    #         HI_tau_.append(HI_tau)
    #         HI_leg_.append(HI_leg)
    #     mpi_barrier()
    #     HI_leg_ = MPI.COMM_WORLD.gather(HI_leg_, root=0)
    #     HI_tau_ = MPI.COMM_WORLD.gather(HI_tau_, root=0)
    #     if whoami == 0:
    #         fname="./db/G_HI_leg.csv"
    #         write_me_MPI_GFs(fname, HI_leg_)
    #         fname="./db/G_HI_tau.csv"
    #         write_me_MPI_GFs(fname, HI_tau_)

    # if solvers["ED_KCL"]: 
    #     if whoami == 0:            
    #         print("*"*20, " Running ED_KCL", "*"*20)
    #         print("")
    #     solv_param["bathsize"] = 4
    #     gt_KCL_ = []
    #     gl_KCL_ = []
    #     for p in params:
    #         print(p)
    #         try:
    #             shutil.rmtree("./ED_RUN/")
    #         except OSError as e:
    #             print ("Error: %s - %s." % (e.filename, e.strerror))
                
    #         ED = ED_KCL_solver(solv_param)
    #         ED.solve_KCL(param=p)
    #         gt,gl = ED.extract_data()
            
    #         gt_KCL_.append(gt)
    #         gl_KCL_.append(gl)
                        
    #     if whoami == 0:
    #         fname="./db/G_ED_KCL_leg.csv"
    #         write_me_SERIAL_ed(fname, gl_KCL_)
    #         fname="./db/G_ED_KCL_tau.csv"
    #         write_me_SERIAL_ed(fname, gt_KCL_)

    # if solvers["ED_POM"]: 
    #     if whoami == 0:            
    #         print("*"*20, " Running ED_POM", "*"*20)
    #         print("")
    #     G_fn = label("G_ED_POM", basis)
    #     G_fn_iw = label("G_ED_POM", "iw")
    #     solv_param["write"] = G_fn
    #     solv_param["write_iw"] = G_fn_iw
    #     delete_files("G_ED_POM", beta)
    #     solv_param["bathsize"] = 4
    #     gtau_POM_ = []
    #     gt_POM_ = []
    #     gl_POM_ = []
    #     for p in params:
    #         ED = ED_POM_solver(solv_param)
    #         ED.solve_dirty(param=p)
    #         gt,gl = ED.extract_data()
    #         gt_POM_.append(gt)
    #         gl_POM_.append(gl)            
    #     if whoami == 0:
    #         fname="./db/G_ED_POM_tau.csv"
    #         write_me_SERIAL_POM_tau(fname, gt_POM_)
    #         fname="./db/G_ED_POM_leg.csv"
    #         write_me_SERIAL_POM(fname, gl_POM_)
                        
     
if __name__ == "__main__":
    input = "config.ini"
    system_param, db_param, aim_param, learn_param = get_config(input)
    solve_AIM(system_param, db_param, aim_param)
