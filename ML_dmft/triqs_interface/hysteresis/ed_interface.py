from ML_dmft.utility.read_yml import solve_db_args
from ML_dmft.utility.tools import read_file_name
from ML_dmft.numpy_GF.fit_hyb import cal_err_semi_circular_DOS_symmetry
from ML_dmft.solvers.ED_KCL_solver import RUN_ED
import numpy as np
import sys,contextlib


def _ed_args_2_fit(aim_settings:dict,
                   ED_min_bath:int,
                   beta:float):
    initial_fit_err =  aim_settings['ed-solver']['fit_err_init']
    max_fit_err_tol = aim_settings['ed-solver']['max_fit_err_tol']
    hyb_fit_method = aim_settings['ed-solver']['fit_method']
    V_bound =  aim_settings['ed-solver']['V_bound']
    e_bound =  aim_settings['ed-solver']['e_bound']
    init_V_bound = aim_settings['ed-solver']['init_V_bound']
    init_e_bound = aim_settings['ed-solver']['init_e_bound']
    err_tol_fix =  aim_settings['ed-solver']['err_tol_fix']
    minimizer_maxiter =  aim_settings['ed-solver']['minimizer_maxiter']
    fit_max_iter  =  aim_settings['ed-solver']['fit_max_iter']
    delta_from_zero =  aim_settings['ed-solver']['delta_from_zero']

    n_iw_cut = 32
    verbose = 0
    fit_semi_circular_dict=dict(num_imp=ED_min_bath,
                                err_tol=initial_fit_err,
                                max_fitting_err=max_fit_err_tol,
                                beta=beta,
                                n_iw=n_iw_cut,
                                W=None,
                                omega_c=32,
                                method=hyb_fit_method,
                                V_bound=V_bound,
                                E_bound=e_bound,
                                V_bound_init=init_V_bound,
                                E_bound_init=init_e_bound,
                                err_tol_fix = err_tol_fix,
                                max_iter = fit_max_iter,
                                minimizer_maxiter = minimizer_maxiter,
                                delta_from_zero = delta_from_zero,
                                disp = False if verbose < 4 else True,
                                )
    return fit_semi_circular_dict

def fit_params():
    with contextlib.redirect_stdout(None):
        _,fitted_erro,Fit_Success,e_list,V_list=cal_err_semi_circular_DOS_symmetry

def ed_interface(args,U,beta,g_iw):
    solver_params={}
    solver_params['gf_param']=args['gf_param']
    solver_params['solve dmft']=args['dmft_param']
    for key in args['ed_params']:
        solver_params['solve dmft'][key]=args['ed_params'][key]
    
    
    #########################################################
    params = {}
    params["U"] = float(U)
    params["eps"] = -float(U)/2
    params['W'] = 1
    N = 7 #TODO link tto
    params['N'] = N
    params["E_p"] = [float(e) for e in np.zeros(N)]
    params["V_p"] = [float(v) for v in np.zeros(N)]

    # ED=RUN_ED(params,solver_params)
    ############################################################