import configparser
import json
import numpy as np
from ML_dmft.utility.mpi_tools import mpi_rank

def get_config(config_file):
    # print("Reading config")
    config = configparser.ConfigParser()  
    config.read(config_file)

    n_samples = config.getint("main settings", "aim_samples")
    mott_num = config.getint("main settings","mott_num")

    verbose = config.getint("development","verbose")

    # system parameters
    beta = config.getfloat("gf settings", "beta")
    tau_file = config.getint("gf settings", "tau_file")
    n_tau = config.getint("gf settings", "n_tau")
    n_iw = config.getint("gf settings", "n_iw")
    n_l = config.getint("gf settings", "n_l")

    # aim parameters
    bethe = config.getboolean("aim settings", "bethe")
    discrete = config.getboolean("aim settings", "discrete")

    # ed parameters
    ED_min_bath =  config.getint("ed solver", "ed_min_bath")
    omp_threads = config.getint("ed solver", "omp_threads")
    ed_fit_param_on = config.getboolean("ed solver",'ed_fit_param_on')

    ed_params={'ED_min_bath':ED_min_bath,
                'omp_threads':omp_threads,
                'ed_fit_param_on':ed_fit_param_on
                }

    #IPT parameter
    hatree_shift_back=config.getboolean("IPT setting", "hatree_shift_back")
    hartree_shift_err_tol=config.getfloat("IPT setting", "hartree_shift_err_tol")
    ipt_param={"hatree_shift_back":hatree_shift_back,
                "hartree_shift_err_tol":hartree_shift_err_tol
            }

    #dmft parameters
    dmft_on = config.getboolean("dmft settings", "dmft_on")
    mixing_parameter = config.getfloat("dmft settings", "mixing_parameter")
    n_iter_max = config.getint("dmft settings", "n_iter_max")
    converge_iter_tol = config.getint("dmft settings", "converge_iter_tol")
    error_tol = config.getfloat("dmft settings", "error_tol")

    #output database format
    save_csv = config.getboolean("output database format", "save_csv")
    save_pkl_dict = config.getboolean("output database format", "save_pkl_dict")
    save_small_files = config.getboolean("output database format", "save_small_files")

    output_format={"save_csv":save_csv,
                    "save_pkl_dict":save_pkl_dict,
                    "save_small_files":save_small_files}

    #discrete_hyb
    discrete_hyb_nk=config.getint("discrete hyb setting", "n_k")

    # database parameters
    half_filled = config.getboolean("database", "half-filled")
    U_0_shift = config.getboolean("database", "U_0_shift")
    database_U = json.loads(config.get("database", "U_" ))
    database_eps = json.loads(config.get("database", "eps_" ))
    database_D = json.loads(config.get("database", "D_" ))
    database_V = json.loads(config.get("database", "V_" ))
    database_N = json.loads(config.get("database", "N_" ))


    if bethe:
        if discrete:
            raise ValueError('either bethe of discrete')
        if mpi_rank() == 0:
            print(20*'#','study bethe lattice',20*'#')

    if discrete:
        if bethe:
            raise ValueError('either bethe of discrete')
        if mpi_rank() == 0:
            print(20*'#','study 2d square lattice with discrete hyb',20*'#')

    # conv string lists to float lists
    U_, eps_, D_, V_, N_=[], [], [], [], []
    [U_.append(float(i)) for i in database_U]
    [eps_.append(float(i)) for i in database_eps]
    [D_.append(float(i)) for i in database_D]
    [V_.append(float(i)) for i in database_V]
    [N_.append(float(i)) for i in database_N]

    if beta > 10.:
        assert n_l > 10, "Please specify n_l > 10. For your choice of beta = " + str(beta) + " and n_l = " + str(n_l) + " is too small"
        # assert n_l < 40, "Please specify n_l < 40. For your choice of beta = " + str(beta) + " and n_l = " + str(n_l) + " will have too many zeros"
    # elif beta < 5:
    #     assert n_l < 10, "Please specify n_l < 10. For your choice of beta = " + str(beta) + " and n_l = " + str(n_l) + " will have too many zeros"
    # n_tau = n_iw*10 + 1 

    assert n_tau > n_iw*10, "Please make sure that you have many more n_tau than n_iw"


    db_param = {"mott_num":mott_num,
                "n_samples": n_samples,
                "U_": U_,
                "eps_": eps_,
                "D_": D_,
                "V_": V_,
                "N_": N_,
                "n_iw": n_iw,
                "n_tau": n_tau,
                "n_l": n_l,
                'half_filled':half_filled,
                'bethe':bethe,
                'discrete':discrete,
                'U_0_shift':U_0_shift,
                }

    gf_params = {
                  "beta": beta,
                  "n_l": n_l,
                  "n_iw": n_iw,
                  "n_tau": n_tau,
                  "indices": [0],
                  "target_n_tau": tau_file,
                  "bethe":bethe,
                  "discrete":discrete,
                  "discrete_hyb_nk":discrete_hyb_nk,
                  "verbose":verbose
                }
    
    dmft_param = {
            "dmft_on":dmft_on,
            "mixing_parameter":mixing_parameter,
            "n_iter_max":n_iter_max,
            "converge_iter_tol":converge_iter_tol,
            "error_tol":error_tol}

    
    args={  'db_param':db_param, 
            'gf_param':gf_params,
            'ipt_param':ipt_param,
            'dmft_param':dmft_param,
            'ed_params':ed_params,
            'output_format':output_format
            }


    return args
