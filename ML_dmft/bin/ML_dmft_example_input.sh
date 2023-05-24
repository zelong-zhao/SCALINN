#!/usr/bin/bash

cat << EOF > example_solver_inputs.yml
main-settings:
    aim_params_file : './db/aim_params.csv'
    out_solver_name : None
    item_idx : -1 #-1 for all of data

gf-settings:
    beta : 10
    tau_file : 512
    n_iw : 1024
    n_tau : 20480
    n_l : 64

aim-settings:
    bethe : False
    discrete : True

ed-solver:
    ed_min_bath : 7
    omp_threads : 1
    ed_fit_param_on : True
    fit_method : 'BFGS' # 'CG' 
    fit_err_init : 1.0e-4 
    max_fit_err_tol : 1.0e-4
    err_tol_fix : True
    delta_from_zero : 1.0e-9 #small shift from zero.
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

cat << EOF > example_gendb_inputs.yml
main-settings:
    aim_samples : 11 #num of samples to generate

gf-settings:
    beta : 10 #only affects semi-fit generate

aim-settings:
    bethe : True 
    discrete : False

mit-db:
    mott_num : 0 #0 to turn off
    U_0_shift : False # True to add a U0 between 0~1 to db U terms, only activate with mott_num>0

semi-fit-settings:
    fit_err_init : 2.81
    max_fit_err_tol : 2.81
    err_tol_fix : True
    fit_method : 'BFGS' # 'CG' 
    delta_from_zero : 1.0e-9 #small shift from zero.
    fit_max_iter : 200 #max iteration for fitting "outerlook"
    minimizer_maxiter : 20 #max iteration for solver
    init_V_bound : [-3,3] #initial bound as guess.
    init_e_bound : [-3,3]

database:
    half-filled : True
    U_ : [0,10]
    eps_ : [-6, 6] 
    D_ : [1, 1]
    V_ : [-5.,5.] 
    e_ : [-5.,5.]
    N_ : 4
EOF