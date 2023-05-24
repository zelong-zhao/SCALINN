from ML_dmft.utility.mpi_tools import mpi_rank
from ML_dmft.utility.tools import dump_to_yaml,read_yaml
import yaml
import numpy as np

def get_config():

    # reading
    if mpi_rank() == 0:
        print('reading config.yml')
        

    config=read_yaml('config.yml')


    n_samples=int(config['main-settings']['aim_samples'])

    mott_num=int(config['mit-db']['mott_num'])
    U_0_shift = bool(config["mit-db"]["U_0_shift"])
    

    verbose = int(config['development']['verbose'])

    # system parameters
    beta=float(config['gf-settings']['beta'])
    tau_file=int(config['gf-settings']['tau_file'])
    n_tau=int(config['gf-settings']['n_tau'])
    n_iw=int(config['gf-settings']['n_iw'])
    n_l=int(config['gf-settings']['n_l'])

    # aim parameters
    bethe = bool(config["aim-settings"]["bethe"])
    discrete = bool(config["aim-settings"]["discrete"])

    # ed parameters
    ED_min_bath=int(config['ed-solver']['ed_min_bath'])
    omp_threads=int(config['ed-solver']['omp_threads'])
    ed_fit_param_on=bool(config['ed-solver']['ed_fit_param_on'])

    ed_params=dict(ED_min_bath=ED_min_bath,
                    omp_threads=omp_threads,
                    ed_fit_param_on=ed_fit_param_on)
    
    #IPT parameter
    hatree_shift_back=bool(config['IPT-setting']['hatree_shift_back'])
    hartree_shift_err_tol=float(config['IPT-setting']['hartree_shift_err_tol'])

    ipt_param=dict(hatree_shift_back=hatree_shift_back,
                hartree_shift_err_tol=hartree_shift_err_tol
                )

    #dmft parameters
    dmft_on = bool(config["dmft-settings"]["dmft_on"])
    mixing_parameter=float(config["dmft-settings"]["mixing_parameter"])
    n_iter_max=int(config["dmft-settings"]["n_iter_max"])
    converge_iter_tol=int(config["dmft-settings"]["converge_iter_tol"])
    error_tol = float(config["dmft-settings"]["error_tol"])

    dmft_param = {
            "dmft_on":dmft_on,
            "mixing_parameter":mixing_parameter,
            "n_iter_max":n_iter_max,
            "converge_iter_tol":converge_iter_tol,
            "error_tol":error_tol}

    #output database format
    save_csv = bool(config["output-database-format"]["save_csv"])
    save_pkl_dict = bool(config["output-database-format"]["save_pkl_dict"])
    save_small_files = bool(config["output-database-format"]["save_small_files"])

    output_format={"save_csv":save_csv,
                    "save_pkl_dict":save_pkl_dict,
                    "save_small_files":save_small_files}


    #discrete_hyb
    discrete_hyb_nk=int(config["discrete-hyb-setting"]["n_k"])

    # database parameters
    half_filled = bool(config["database"]["half-filled"])
    U_ = [float(i) for i in config["database"]["U_"]]
    eps_ = [float(i) for i in config["database"]["eps_"]]
    D_ = [float(i) for i in config["database"]["D_"]]
    V_ = [float(i) for i in config["database"]["V_"]]
    e_ = [float(i) for i in config["database"]["e_"]]
    N_ = int(config["database"]["N_"])
    fit_err_tol = float(config["database"]["fit_err_tol"])
    delta_from_zero = float(config["database"]["delta_from_zero"])

    if bethe:
        if discrete:
            raise ValueError('either bethe of discrete')
        if mpi_rank() == 0:
            print(20*'#','study bethe lattice',20*'#')
    else:
        if not discrete:
            raise ValueError('either bethe of discrete')

        if discrete:
            if mpi_rank() == 0:
                print(20*'#','study 2d square lattice with discrete hyb',20*'#')


    if beta > 10.:
        assert n_l > 10, "Please specify n_l > 10. For your choice of beta = " + str(beta) + " and n_l = " + str(n_l) + " is too small"

    assert n_tau > n_iw*10, "Please make sure that you have many more n_tau than n_iw"

    db_param = {"mott_num":mott_num,
                "n_samples": n_samples,
                "U_": U_,
                "eps_": eps_,
                "D_": D_,
                "V_": V_,
                "e_": e_,
                "N_": N_,
                "n_iw": n_iw,
                "n_tau": n_tau,
                "n_l": n_l,
                'half_filled':half_filled,
                'bethe':bethe,
                'discrete':discrete,
                'U_0_shift':U_0_shift,
                'beta':beta,
                'fit_err_tol':fit_err_tol,
                'delta_from_zero':delta_from_zero,
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

    # aim parameters
    bethe = bool(config["aim-settings"]["bethe"])
    discrete = bool(config["aim-settings"]["discrete"])
    
    args={  'db_param':db_param, 
            'gf_param':gf_params,
            'ipt_param':ipt_param,
            'dmft_param':dmft_param,
            'ed_params':ed_params,
            'output_format':output_format
            }

    if mpi_rank() == 0:
        dump_to_yaml('out_config_read.yml',args)

    return args


def solve_db_args(file):
    if mpi_rank() == 0:
        print(f'reading {file}')

    config=read_yaml(file)

    aim_params_file = str(config['main-settings']['aim_params_file'])
    out_solver_name = str(config['main-settings']['out_solver_name'])
    if out_solver_name.lower() in ['none','null','default']:
        out_solver_name = None

    item_idx = str(config['main-settings']['item_idx'])
    if item_idx in ["-1","None","all"]: 
        item_idx = None
    else:
        item_idx = int(item_idx)

    verbose = int(config['development']['verbose'])
                   
    # system parameters
    beta=float(config['gf-settings']['beta'])
    tau_file=int(config['gf-settings']['tau_file'])
    n_tau=int(config['gf-settings']['n_tau'])
    n_iw=int(config['gf-settings']['n_iw'])
    n_l=int(config['gf-settings']['n_l'])

    # aim parameters
    bethe = bool(config["aim-settings"]["bethe"])
    discrete = bool(config["aim-settings"]["discrete"])

    # ed parameters
    ED_min_bath=int(config['ed-solver']['ed_min_bath'])
    omp_threads=int(config['ed-solver']['omp_threads'])
    ed_fit_param_on=bool(config['ed-solver']['ed_fit_param_on'])

    hyb_fit_method=str(config['ed-solver']['fit_method'])
    fit_err_tol=float(config['ed-solver']['fit_err_init'])
    max_fit_err_tol=float(config['ed-solver']['max_fit_err_tol'])
    fit_max_iter=int(config['ed-solver']['fit_max_iter'])
    minimizer_maxiter = int(config['ed-solver']['minimizer_maxiter'])
    delta_from_zero = float(config["ed-solver"]["delta_from_zero"])
    err_tol_fix = bool(config["ed-solver"]["err_tol_fix"])
    V_bound = [float(i) for i in config["ed-solver"]["V_bound"]]
    e_bound = [float(i) for i in config["ed-solver"]["e_bound"]]
    init_V_bound = [float(i) for i in config["ed-solver"]["init_V_bound"]]
    init_e_bound = [float(i) for i in config["ed-solver"]["init_e_bound"]]


    ed_params=dict(ED_min_bath=ED_min_bath,
                    omp_threads=omp_threads,
                    ed_fit_param_on=ed_fit_param_on,
                    hyb_fit_method=hyb_fit_method,
                    V_bound=V_bound,
                    e_bound=e_bound,
                    fit_err_tol=fit_err_tol,
                    max_fit_err_tol=max_fit_err_tol,
                    fit_max_iter=fit_max_iter,
                    delta_from_zero=delta_from_zero,
                    err_tol_fix=err_tol_fix,
                    init_V_bound=init_V_bound,
                    init_e_bound=init_e_bound,
                    minimizer_maxiter=minimizer_maxiter
                    )
    
    #IPT parameter
    hatree_shift_back=bool(config['IPT-setting']['hatree_shift_back'])
    hartree_shift_err_tol=float(config['IPT-setting']['hartree_shift_err_tol'])

    ipt_param=dict(hatree_shift_back=hatree_shift_back,
                hartree_shift_err_tol=hartree_shift_err_tol
                )

    #dmft parameters
    dmft_on = bool(config["dmft-settings"]["dmft_on"])
    mixing_parameter=float(config["dmft-settings"]["mixing_parameter"])
    n_iter_max=int(config["dmft-settings"]["n_iter_max"])
    converge_iter_tol=int(config["dmft-settings"]["converge_iter_tol"])
    error_tol = float(config["dmft-settings"]["error_tol"])

    dmft_param = {
            "dmft_on":dmft_on,
            "mixing_parameter":mixing_parameter,
            "n_iter_max":n_iter_max,
            "converge_iter_tol":converge_iter_tol,
            "error_tol":error_tol}


    output_format={"save_csv":False,
                    "save_pkl_dict":False,
                    "save_small_files":True}


    #discrete_hyb
    discrete_hyb_nk=int(config["discrete-hyb-setting"]["n_k"])

    assert len(set([bethe,discrete])) == 2,'must one correct on wrong'

    if mpi_rank() == 0:
        if bethe: 
            print(20*'#','study bethe lattice',20*'#')
        elif discrete: 
            print(20*'#','study 2d square lattice with discrete hyb',20*'#')

    if beta > 10.:
        assert n_l > 10, "Please specify n_l > 10. For your choice of beta = " + str(beta) + " and n_l = " + str(n_l) + " is too small"

    assert n_tau > n_iw*10, "Please make sure that you have many more n_tau than n_iw"


    gf_params = {"beta": beta,
                  "n_l": n_l,
                  "n_iw": n_iw,
                  "n_tau": n_tau,
                  "indices": [0],
                  "target_n_tau": tau_file,
                  "bethe":bethe,
                  "discrete":discrete,
                  "discrete_hyb_nk":discrete_hyb_nk,
                  "verbose":verbose}

    # aim parameters
    bethe = bool(config["aim-settings"]["bethe"])
    discrete = bool(config["aim-settings"]["discrete"])
    
    args={  'main_params':dict(aim_params_file=aim_params_file,item_idx=item_idx,out_solver_name=out_solver_name),
            'gf_param':gf_params,
            'ipt_param':ipt_param,
            'dmft_param':dmft_param,
            'ed_params':ed_params,
            'output_format':output_format
            }
    return args

def gen_db_args(file):
    if mpi_rank() == 0:
        print(f'reading {file}')

    config=read_yaml(file)

    # database parameters
    half_filled = bool(config["database"]["half-filled"])
    U_ = [float(i) for i in config["database"]["U_"]]
    eps_ = [float(i) for i in config["database"]["eps_"]]
    D_ = [float(i) for i in config["database"]["D_"]]
    V_ = [float(i) for i in config["database"]["V_"]]
    e_ = [float(i) for i in config["database"]["e_"]]
    N_ = int(config["database"]["N_"])

    hyb_fit_method=str(config['semi-fit-settings']['fit_method'])
    fit_err_tol=float(config['semi-fit-settings']['fit_err_init'])
    max_fit_err_tol=float(config['semi-fit-settings']['max_fit_err_tol'])
    fit_max_iter=int(config['semi-fit-settings']['fit_max_iter'])
    minimizer_maxiter = int(config['semi-fit-settings']['minimizer_maxiter'])
    delta_from_zero = float(config["semi-fit-settings"]["delta_from_zero"])
    err_tol_fix = bool(config["semi-fit-settings"]["err_tol_fix"])
    init_V_bound = [float(i) for i in config["semi-fit-settings"]["init_V_bound"]]
    init_e_bound = [float(i) for i in config["semi-fit-settings"]["init_e_bound"]]

    n_samples=int(config['main-settings']['aim_samples'])

    bethe = bool(config["aim-settings"]["bethe"])
    discrete = bool(config["aim-settings"]["discrete"])

    assert len(set([bethe,discrete])) == 2,'must one correct on wrong'
    if mpi_rank() == 0:
        if bethe: 
            print(20*'#','study bethe lattice',20*'#')
        elif discrete: 
            print(20*'#','study 2d square lattice with discrete hyb',20*'#')

    mott_num=int(config['mit-db']['mott_num'])
    U_0_shift = bool(config["mit-db"]["U_0_shift"])

    # system parameters
    beta=float(config['gf-settings']['beta'])


    db_param = {"mott_num":mott_num,
                "n_samples": n_samples,
                "U_": U_,
                "eps_": eps_,
                "D_": D_,
                "V_": V_,
                "e_": e_,
                "N_": N_,
                'half_filled':half_filled,
                'bethe':bethe,
                'discrete':discrete,
                'U_0_shift':U_0_shift,
                'beta':beta,
                'fit_err_tol':fit_err_tol,
                'delta_from_zero':delta_from_zero,
                'fit_method':hyb_fit_method,
                'fit_max_iter':fit_max_iter,
                'max_fit_err_tol':max_fit_err_tol,
                'minimizer_maxiter':minimizer_maxiter,
                'err_tol_fix':err_tol_fix,
                'init_V_bound':init_V_bound,
                'init_e_bound':init_e_bound,
                }
    
    args=dict(db_param=db_param)
    return args



if __name__ == '__main__':
    args=get_config()