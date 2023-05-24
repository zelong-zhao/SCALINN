import time
from ML_dmft.solvers.ED_KCL_solver import ED_KCL_solver
from ML_dmft.utility.tools import read_params,save2pkl
from ML_dmft.utility.read_config import get_config

def solve_AIM(system_param, db_param):    
    """
    Main routine for solving AIM for the database
    """
    time_begin=time.time()

    beta = system_param["beta"]
    omp_threads = system_param["omp_threads"]
    n_l = db_param["n_l"]
    n_iw = db_param["n_iw"]
    n_tau = db_param["n_tau"]
    target_n_tau = system_param["tau_file"] 
    ED_min_bath =  system_param["ED_min_bath"] 

    indices = [0]
    
    solv_param = {"beta": beta,
                  "n_l": n_l,
                  "n_iw": n_iw,
                  "n_tau": n_tau,
                  "indices": indices,
                  "target_n_tau": target_n_tau,
                  "omp_threads": omp_threads,
                  "ED_min_bath":ED_min_bath
                  }
    
    params = read_params("./db/aim_params.csv")

    time_begin=time.time()
    print("*"*20, " Running ED_KCL", "*"*20)
    print("")
    
    solv_param["bathsize"] = 4
    all_KCL_list = []
    for idx,p in enumerate(params):
            
        ED = ED_KCL_solver(solv_param)
        ED.solve_KCL(param=p)
        all_KCL_list.append(ED.saveGF2mydataformat(param=p))

        if idx%1000 == 0:
            print('total number of points: {}, calculating{}'.format(len(params),idx))

    time_finish=time.time()-time_begin
    print('*'*20,time_finish,'*'*20)
    save2pkl(input_list=all_KCL_list,fname='./db/ED_KCL.pkl')
                        
if __name__ == "__main__":
    input = "config.ini"
    system_param, db_param = get_config(input)
    solve_AIM(system_param, db_param)