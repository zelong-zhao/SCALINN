#!/usr/bin/bash

check_integer(){
   re='^[0-9]+$'
   if ! [[ $1 =~ $re ]] ; then
   echo "error: Not a number" >&2; exit 1
fi
}

check_var() {
  if [ -z "$1" ]; then
    echo "$1 is empty, exiting script with error"
    exit 1
fi
}

example_truncate_ed_tmp(){
check_integer $1
check_integer $2
check_var $3
check_var $4
check_var $5

local bath_site=$1
local beta=$2
local in_param_csv_file=$3
local out_solver_name=$4
local out_solv_inputs_file=$5

if [ -d "$out_solv_inputs_file" ] ; then
    exit 0
fi

cat << EOF > $out_solv_inputs_file
main-settings:
    aim_params_file : './db/$in_param_csv_file'
    out_solver_name : '$out_solver_name'
    item_idx : -1 #-1 for all of data

gf-settings:
    beta : $beta
    tau_file : 512
    n_iw : 1024
    n_tau : 20480
    n_l : 64

aim-settings:
    bethe : False
    discrete : True

ed-solver:
    ed_min_bath : $bath_site
    omp_threads : 1
    ed_fit_param_on : True
    fit_method : 'BFGS' # 'CG' 
    fit_err_init : 1.0e-4 
    max_fit_err_tol : 1.0e-4
    err_tol_fix : True
    delta_from_zero : 1.0e-4 #small shift from zero.
    fit_max_iter : 10 #max iteration for fitting "outerlook"
    minimizer_maxiter : 200 #max iteration for solver
    V_bound : [-3,3] #bound
    e_bound : [-3,3] #bound
    init_V_bound : [-2,2] #initial bound as guess.
    init_e_bound : [-2,2]

dmft-settings:
    dmft_on : False
    mixing_parameter : 0.8
    n_iter_max : 200
    converge_iter_tol : 150
    error_tol : 1.0e-3

IPT-setting:
    hatree_shift_back : True
    hartree_shift_err_tol : 0.5

discrete-hyb-setting:
    n_k : 8

development:
    verbose : 0
EOF
}
