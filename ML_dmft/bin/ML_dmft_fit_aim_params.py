#! /usr/bin/env python
from ML_dmft.utility.read_yml import gen_db_args
from ML_dmft.numpy_GF.fit_hyb import cal_err_semi_circular_DOS_symmetry
import ML_dmft.numpy_GF.GF as GF
import numpy as np
from ML_dmft.utility.tools import read_params,flat_aim
import argparse

def read_file_name():
    parser = argparse.ArgumentParser(description='ML dmft inputs')
    parser.add_argument('--file','-f','-inp','-i',type=str, metavar='f',
                    help='file name must present',required=True) 
    parser.add_argument('--out-file','--out','-out','-o',type=str, metavar='f',
                    help='file name must present',required=True)
    args = parser.parse_args()
    return args

in_file = read_file_name().file
out_file = read_file_name().out_file

args=gen_db_args(in_file)['db_param']

beta = args['beta']
V_bound = args['V_']
e_bound = args['e_']
num_imp = args['N_']


fit_method = args['fit_method']
fit_err_tol = args['fit_err_tol']
fit_max_iter = args['fit_max_iter']
max_fit_err_tol = args['max_fit_err_tol']
minimizer_maxiter = args['minimizer_maxiter']
err_tol_fix = args['err_tol_fix']
delta_from_zero = args['delta_from_zero']
init_V_bound = args['init_V_bound']
init_e_bound = args['init_e_bound']



W=args['D_'][0]
n_iw = 64

in_file = './db/aim_params.csv'
outfile = f'./db/{out_file}'

print(f"In fitting hyb {num_imp=}, \nwill read {in_file=} and write to \n{outfile=}")
print(f"{50*'#'}")
params = read_params(in_file)

assert params[0]['N'] != num_imp, 'readuce bath sites'

for idx,param in enumerate(params):
    print(param)
    onsite,hopping = param['E_p'],param['V_p']
    fit_function = GF.hyb_np(onsite,hopping,n_iw,beta)

    fit_semi_circular_dict=dict(fit_function=fit_function,
                                num_imp=num_imp,
                                beta=beta,
                                err_tol=fit_err_tol,
                                max_fitting_err=max_fit_err_tol,
                                delta_from_zero=delta_from_zero,
                                V_bound = V_bound,
                                W=None,
                                E_bound = e_bound,
                                V_bound_init = init_V_bound,
                                E_bound_init = init_e_bound,
                                minimizer_maxiter=minimizer_maxiter,
                                n_iw=64,
                                omega_c=32,
                                err_tol_fix=err_tol_fix,
                                method=fit_method,
                                max_iter=fit_max_iter,
                                disp=False
                                )

    _,_,_,e_list,V_list=cal_err_semi_circular_DOS_symmetry(**fit_semi_circular_dict)
    params[idx]['N'],params[idx]['E_p'],params[idx]['V_p']=num_imp,e_list,V_list

params = [flat_aim(item)[0] for item in params]
np.savetxt(outfile, params, delimiter=",", fmt="%1.6f")