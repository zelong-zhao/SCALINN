# -*- coding: utf-8 -*-
"""
ML DMFT
===============================
# Author: Even Evan Sheridan 2021
# Author: Zelong Zhao 2022
"""

import numpy as np
from ML_dmft.numpy_GF.fit_hyb import cal_err_semi_circular_DOS_symmetry
from triqs.gf import * #GfImFreq,GfImTime,GfLegendre,Fourier,MatsubaraToLegendre,inverse
import os, shutil, subprocess,sys
import ML_dmft.numpy_GF.GF as GF
from ML_dmft.triqs_interface.v2 import triqs_interface,hyb,get_gf_xy
from numpy import cos
from triqs.lattice import BravaisLattice, BrillouinZone
from sklearn.metrics import mean_squared_error
from ML_dmft.utility.tools import flat_aim,EarlyStopper_DMFT
from time import perf_counter
from ML_dmft.utility.mpi_tools import mpi_rank
from copy import deepcopy
import warnings,tempfile,yaml
import contextlib
from ML_dmft.utility.dmft_performance_checker import Monitor_Dmft_Performance


def ED_threads(param,solver_params):

    rank=mpi_rank()
    
    path=os.path.join(os.getcwd(),'mpi_'+str(rank))
    # print(path)

    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

    os.chdir(path)

    ED = ED_KCL_solver(solver_params['gf_param'])
    ED.solve_KCL_DMFT(param=param,dmft_param=solver_params['solve dmft'])
    out_dict = ED.saveGF2mydataformat(param)

    os.chdir("../")
    shutil.rmtree(path)
    return out_dict


def RUN_ED(param,solver_params):

    rank=mpi_rank()
    current_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp(dir=current_dir)
    path=os.path.join(temp_dir,'mpi_'+str(rank))
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    os.chdir(path)
    ED = ED_KCL_solver(solver_params['gf_param'])
    try:
        ED.solve_KCL_DMFT(param=param,dmft_param=solver_params['solve dmft'],CWD=current_dir)
    except FileNotFoundError:
        warnings.warn(f'{temp_dir} fail to read ED')
        out_solver_param = deepcopy(solver_params['solve dmft'])
        out_solver_param['ed_fit_param_on'] = False
        ED.solve_KCL_DMFT(param=param,dmft_param=out_solver_param)

    os.chdir(current_dir)
    shutil.rmtree(temp_dir)
    return ED

def RUN_ED_with_param_only(param:dict,beta:float,ED_min_bath:int):
    ED_solver_params=f"""

gf_param:
    beta: {beta}
    bethe: false
    discrete: true
    discrete_hyb_nk: 8
    indices:
    - 0
    n_iw: 1024
    n_l: 64
    n_tau: 20480
    target_n_tau: 512
    verbose: 0
solve dmft:
    ED_min_bath: {ED_min_bath}
    converge_iter_tol: 150
    dmft_on: false
    ed_fit_param_on : True
    error_tol: 0.001
    mixing_parameter: 0.8
    n_iter_max: 10
    omp_threads: 1
    hyb_fit_method : 'BFGS' # 'CG' 
    fit_err_tol : 1.0e-4 
    max_fit_err_tol : 1.0e-4
    err_tol_fix : True
    delta_from_zero : 1.0e-9 #small shift from zero.
    fit_max_iter : 10 #max iteration for fitting "outerlook"
    minimizer_maxiter : 200 #max iteration for solver
    V_bound : [-3,3] #bound
    e_bound : [-3,3] #bound
    init_V_bound : [-2,2] #initial bound as guess.
    init_e_bound : [-2,2]
"""
    solver_params=yaml.safe_load(ED_solver_params)

    temp_dir = tempfile.mkdtemp(dir=os.getcwd())
    os.chdir(temp_dir)

    ED = ED_KCL_solver(solver_params['gf_param'])
    ED.solve_KCL_DMFT(param=param,dmft_param=solver_params['solve dmft'])

    os.chdir("../")
    shutil.rmtree(temp_dir)
    # returns G only
    G_iw = ED.G_iw.data[:,:,0][ED.n_iw:2*ED.n_iw]
    
    return G_iw




class ED_KCL_solver():
    """
    Class for the ED pomerol solver
    """

    def __init__(self, solver_params):
        """
        Initialise the PT_solver
        """

        # general
        self.beta = solver_params["beta"]
        self.n_l = solver_params["n_l"]
        self.n_iw = solver_params["n_iw"]
        self.n_tau = solver_params["n_tau"]
        self.indices = solver_params["indices"]
        self.target_n_tau = solver_params["target_n_tau"]

        self.bethe = solver_params["bethe"]
        self.discrete = solver_params["discrete"]
        self.verbose = solver_params['verbose']

        # Matsubara
        self.Delta_iw = GfImFreq(indices=self.indices,
                                 beta=self.beta,
                                 n_points=self.n_iw)
        self.G_iw = self.Delta_iw.copy()
        self.G_0_iw = self.Delta_iw.copy()
        self.Sigma_iw = self.Delta_iw.copy()

        # Imaginary time
        self.G_tau_prev = GfImTime(indices=self.indices,
                                     beta=self.beta,
                                     n_points=self.n_tau)
        self.G_tau = self.G_tau_prev.copy()
        self.imtime_target = np.linspace(0, self.beta,
                                         self.target_n_tau, endpoint=True)

        # Legendre
        self.G_l = GfLegendre(indices = self.indices, beta = self.beta,
                                n_points = self.n_l)

        self.Sigma_tau = self.G_tau.copy()
        self.legendre_target = np.linspace(1, self.n_l, self.n_l, endpoint = True)

        if self.discrete:
            BL = BravaisLattice([(1, 0, 0), (0, 1, 0)]) # Two unit vectors in R3
            BZ = BrillouinZone(BL)
            self.n_k=solver_params['discrete_hyb_nk']
            self.kmesh = MeshBrillouinZone(BZ, n_k=self.n_k)
            if self.verbose >=2:
                print('2d square lattice k points',self.n_k**2)

    def solve_KCL_DMFT(self, param,dmft_param,CWD='./'):
        """
        Solve the AIM using ED
        """
        self.threads = dmft_param["omp_threads"]

        self.ed_fit_param_on=dmft_param['ed_fit_param_on']
        self.ED_min_bath = dmft_param["ED_min_bath"]
        self.V_bound =  dmft_param["V_bound"]
        self.e_bound = dmft_param["e_bound"]
        self.init_V_bound = dmft_param["init_V_bound"]
        self.init_e_bound = dmft_param["init_e_bound"]
        self.hyb_fit_method = dmft_param["hyb_fit_method"]
        self.fit_err_tol = dmft_param["fit_err_tol"]
        initial_fit_err = self.fit_err_tol 
        self.max_fit_err_tol = dmft_param["max_fit_err_tol"]
        self.fit_max_iter = dmft_param["fit_max_iter"]
        self.delta_from_zero = dmft_param["delta_from_zero"]
        self.err_tol_fix =  dmft_param["err_tol_fix"]
        self.minimizer_maxiter = dmft_param["minimizer_maxiter"]
    

        self.param=deepcopy(param)
        self.CWD=CWD

        self.U = param["U"]
        self.eps = param["eps"]
        self.mu = -self.eps-self.U/2

        if self.bethe:
            self.W = param["W"]
            self.e_list = param["E_p"]
            self.V_list = param["V_p"]

        elif self.discrete:
            self.e_list = param["E_p"]
            self.V_list = param["V_p"]
        else :
            raise ValueError('bethe or discrete not defined')
        assert self.ED_min_bath == len(self.e_list), 'Min bath should Equal to num Bath in param'

        if self.verbose >= 2:
                print(f"{self.mu-self.eps=} {self.U=}")
                if np.abs(-self.eps*2-self.U) > 1e-5:
                        print('not doing half-filled!!')

        #dmft params
        self.self_consistent = dmft_param['dmft_on']
        tol = dmft_param['error_tol']
        n_iter_max = dmft_param['n_iter_max']
        dmft_tolerance = dmft_param['converge_iter_tol']
        alpha=dmft_param['mixing_parameter']

        plot_converge = False
        ES=EarlyStopper_DMFT(dmft_tolerance,0)
        if self.self_consistent:
            if self.verbose >=2: print('Running dmft')
            if self.verbose >=4: plot_converge=True
            if plot_converge: 
                DMFT_Monitor = Monitor_Dmft_Performance(f"U_{int(param['U'])}",CWD=self.CWD)
                DMFT_Monitor.init_record_hyb_param()
        else:
            if self.verbose >=2: print('one iteration only')

        # err_array = np.zeros(dmft_tolerance)

        if self.discrete: # if self.beta?
            self.Delta_iw.data[:,0,0]=np.zeros(2*self.n_iw)
            self.Delta_iw << hyb(param["E_p"], param["V_p"]) 

        if self.bethe:
            if self.ed_fit_param_on:
                self.Delta_iw <<  hyb(param["E_p"], param["V_p"])			
            else:
                self.Delta_iw <<  ((self.W**2)/4.) * SemiCircular(self.W) 
        
        if self.discrete and self.self_consistent: raise ValueError('bethe dmft only')
 
        for iter_num in range(n_iter_max):
            self.G_0_iw << inverse(iOmega_n +self.mu-self.eps - self.Delta_iw)

            ced_ed_path="./ED_RUN"
            create_ced_ed_inputs_common(ced_ed_path=ced_ed_path
                    ,beta=self.beta
                    ,U=self.U
                    ,eps=self.eps
                    ,bath_size=self.ED_min_bath,
                    n_iw=self.n_iw,
                    mu=self.mu,
                    )
            #IMPOSE hybridisation
            if not self.ed_fit_param_on:
                if self.verbose >= 2:	print('not using ed fit param!!!')
                mesh,delta=get_gf_xy(self.Delta_iw)
                mesh=mesh.imag[self.n_iw:]
                delta=delta[self.n_iw:]
                out_data=[mesh.flatten(),delta.real.flatten(),delta.imag.flatten()]
                out_data=np.array(out_data).T

                np.savetxt("./ED_RUN/delta_input1",out_data ,delimiter=" ")
                np.savetxt("./ED_RUN/delta_input2",out_data ,delimiter=" ")

            #IMPOSE hybridisation discrete
            else:
                if self.bethe:
                    if self.self_consistent:
                        DMFT_Monitor.record_hyb_param(flat_aim(self.param))
                        n_iw_cut = 64
                        delta_iw = self.Delta_iw.data[:,:,0][self.n_iw:]
                        delta_iw = delta_iw[:n_iw_cut]
                        time_start_fit = perf_counter()
                        fit_semi_circular_dict=dict(num_imp=self.ED_min_bath,
                                                    err_tol=initial_fit_err,
                                                    max_fitting_err=self.max_fit_err_tol,
                                                    beta=self.beta,
                                                    n_iw=n_iw_cut,
                                                    W=None,
                                                    fit_function=delta_iw,
                                                    omega_c=32,
                                                    method=self.hyb_fit_method,
                                                    V_bound=self.V_bound,
                                                    E_bound=self.e_bound,
                                                    V_bound_init=self.init_V_bound,
                                                    E_bound_init=self.init_e_bound,
                                                    err_tol_fix = self.err_tol_fix,
                                                    max_iter = self.fit_max_iter,
                                                    minimizer_maxiter = self.minimizer_maxiter,
                                                    delta_from_zero = self.delta_from_zero,
                                                    disp = False if self.verbose < 4 else True,
                                                    )
                        where2print = sys.stdout if self.verbose >= 3 else None
                        with contextlib.redirect_stdout(where2print):
                            _,fitted_erro,Fit_Success,self.e_list,self.V_list=cal_err_semi_circular_DOS_symmetry(**fit_semi_circular_dict)
                            self.param["E_p"],self.param["V_p"]=self.e_list,self.V_list
                        # initial_fit_err=max(fitted_erro/10.,self.fit_err_tol)
                        if self.verbose >=2 : print(f"{self.param=}")
                        if self.verbose >=2 : print(f"In ED-solver: {initial_fit_err=:.5e}")
                        if self.verbose >=2 : print(f"IN ED-solver: {fitted_erro=:.5e} {Fit_Success=}")
                        if self.verbose >=2 : print(f"IN ED-solver: Time to Finish Hyb update={perf_counter()-time_start_fit:.5f}s")

                x_arr = []
                if self.verbose >= 2:
                    print('IN ED-solver: Discrete hyb updating')
                    print(f"{self.e_list=}\n{self.V_list=}")
                self.bath_ced=format_bath_for_ced_ed(self.e_list,self.V_list)
                np.savetxt("./ED_RUN/ed.fit.param", self.bath_ced, newline=' ', delimiter=" ")
                np.savetxt("./ED_RUN/ed.skip.fit", x_arr, newline=' ', delimiter=" ")

            # run solver to find G_imp,Sigma_imp
            omp_threads=self.threads
            # run ED and
            time_start_run_ED = perf_counter()
            try:
                g,sig=run_ced_ed_solver(ced_ed_path, omp_threads)
            except FileExistsError:
                print(f"{param=}")
                raise FileExistsError('ED did not finished properly')
            if self.verbose >= 2 : print(f"IN ED-solver: Time to Finish ED={perf_counter()-time_start_run_ED:.5f}s")

            g=gf_to_triqs(g,self.beta,self.n_iw,"G")
            sig=gf_to_triqs(sig,self.beta,self.n_iw,"Sigma")

            self.G_iw << g
            self.G_tau << Fourier(self.G_iw)
            self.G_l << MatsubaraToLegendre(self.G_iw)
            self.Sigma_iw << sig
            self.Z=1 / (1 - (sig(0)[0,0].imag * self.beta / np.pi))
            self.logfile_p1=logfilep1()
            # self.G_iw << inverse(inverse(self.G_0_iw) - self.Sigma_iw) #dyson find G_iw

            if not self.self_consistent:
                if self.verbose >= 1:
                    print("running 1 iteration only")
                break
        

            err=max(np.abs(self.G_tau_prev.data[:,0,0].real-self.G_tau.data[:,0,0].real))

            if err < tol and iter_num >= 1:
                if self.verbose >= 1: print("err max {}".format(err))
                print(f"!ED-Solver {iter_num=}, {err=:.5e}, {tol=:.5e}. Calculation Converged \n{50*'#'}")
                break
            # err_array[int(iter_num%dmft_tolerance)]=err

            self.G_tau_prev << self.G_tau

            if self.verbose >= 2:
                print(f"In ED-Solver: Iter{iter_num=} G_tau change err{err:.5e} and {tol=:.5e} \n")

            if  iter_num > dmft_tolerance and ES.early_stop(err):
                if self.verbose >= 1: print("U={} Fail to converge {}".format(self.U,iter_num))
                if self.verbose >= 1: print("err max {}".format(err))
                if self.verbose >= 1: print('err mse',mean_squared_error(self.G_tau_prev.data[:,0,0].real,self.G_tau.data[:,0,0].real))
                raise SystemError('Error')
            
            if plot_converge:
                DMFT_Monitor.init_oneframe()
                DMFT_Monitor.plot_oneframe(GF.matsubara_freq(self.beta,32),self.Delta_iw.data[:,:,0][self.n_iw:],'delta')
                DMFT_Monitor.plot_oneframe(GF.matsubara_freq(self.beta,32),self.G_iw.data[:,:,0][self.n_iw:],'ED')
                DMFT_Monitor.finish_oneframe()

            if self.discrete:
                if self.verbose >= 2:
                    print('discrete hyb updating')
                Delta_new_iw = self.Delta_iw.copy()
                Delta_new_iw << self.update_hyb()
                self.Delta_iw << alpha * Delta_new_iw + (1-alpha) * self.Delta_iw
            elif self.bethe:
                Delta_new_iw = self.Delta_iw.copy()
                Delta_new_iw << ((self.W**2)/4.)*self.G_iw
                self.Delta_iw << alpha * Delta_new_iw + (1-alpha) * self.Delta_iw
            else:
                raise ValueError('define the lattice type properly')

            if iter_num == n_iter_max-1:
                print('err ',mean_squared_error(self.G_tau_prev.data[:,0,0].real,self.G_tau.data[:,0,0].real))
                print("Solution not converged! after %d"%iter_num)

        if plot_converge: DMFT_Monitor.finish_allframe()

    def update_hyb(self):
        Delta_new_iw = GfImFreq(indices=self.indices,
                beta=self.beta,
                n_points=self.n_iw)

        for k in self.kmesh:
            Delta_new_iw << Delta_new_iw + inverse(iOmega_n+self.mu-self.epsi(k)-self.Sigma_iw)
        Delta_new_iw << Delta_new_iw/(self.n_k**2)
        Delta_new_iw << iOmega_n + self.mu - self.Sigma_iw - inverse(Delta_new_iw)
        return Delta_new_iw


    # def err_converged(self,err_array,tol):
    # 	for item in err_array:
    # 		if item > err_array[0]+tol or item < err_array[0]-tol :
    # 			if self.verbose >= 3: print(item,err_array[0],tol)
    # 			return False
    # 	return True


    def epsi(self,k):
        self.t=self.eps-self.mu
        return -2 *  self.t * (cos(k[0]) + cos(k[1]))

    def extract_data(self):
        data_tau = np.asarray(get_data_basis(self.G_tau, "tau"))

        idxs = np.linspace(0, len(data_tau[0]) - 1, self.target_n_tau).astype(int)
        new_tau=np.zeros((1,len(idxs)))
        for i,idx in enumerate(idxs):
            new_tau[0][i]=data_tau[1][idx]

        data_leg = np.asarray(get_data_basis(self.G_l, "legendre"))
        ret_leg = data_leg[1]
        ret_leg = ret_leg.reshape((1,1,len(ret_leg)))
        return new_tau, ret_leg

    def extract_data_G0(self):
        G_0_tau=self.G_tau.copy()
        G_0_tau << Fourier(self.G_0_iw)

        G_0_l = self.G_l.copy()
        G_0_l << MatsubaraToLegendre(self.G_0_iw)

        data_tau = np.asarray(get_data_basis(G_0_tau, "tau"))

        idxs = np.linspace(0, len(data_tau[0]) - 1, self.target_n_tau).astype(int)
        new_tau=np.zeros((1,len(idxs)))
        for i,idx in enumerate(idxs):
            new_tau[0][i]=data_tau[1][idx]

        data_leg = np.asarray(get_data_basis(G_0_l, "legendre"))
        ret_leg = data_leg[1]
        ret_leg = ret_leg.reshape((1,1,len(ret_leg)))
        return new_tau, ret_leg

    def saveGF2mydataformat(self,param):
        """
        the dictionary should include
        1) AIM params
        2) data info beta n_l n_tau, n_iw
        3) G_l
        4) G_tau
        """
        G_tau,G_l=self.extract_data()
        G_tau=np.asarray(G_tau).flatten()
        G_l=np.asarray(G_l).flatten()

        data_info={'beta':self.beta,
                    'n_l':self.n_l,
                    'n_iw':self.n_iw,
                    'n_tau':self.target_n_tau
                }
        param_out=deepcopy(param)

        dat_struct={'AIM params':param_out,
                    'data info':data_info,
                    'G_tau':G_tau,
                    'G_l':G_l,
                    'Sigma_iw':self.Sigma_iw.data[:,:,0].T,
                    'G0_iw':self.G_0_iw.data[:,:,0].T,
                    'G_iw':self.G_iw.data[:,:,0].T,
                    'logfile-p1_DC':self.logfile_p1,
                    'Z':self.Z
                    }

        return dat_struct

    def save_as_small_files(self,param,root):
        """
        save to ./db/ED_KCL_4/1/files...
        """
        if not os.path.isdir(root):
                os.makedirs(root)

        G_tau,G_l=self.extract_data()
        G_tau=np.asarray(G_tau).flatten()
        G_l=np.asarray(G_l).flatten()
        iw_obj={'Sigma_iw':self.Sigma_iw.data[:,:,0],
                    'G0_iw':self.G_0_iw.data[:,:,0],
                    'G_iw':self.G_iw.data[:,:,0]
                    }

        for key in iw_obj:
                np.savetxt(os.path.join(root,f'{key}.dat'),iw_obj[key][self.n_iw:2*self.n_iw],fmt='%20.10e \t %20.10e',delimiter='\t')

        param_out = deepcopy(param)
        param_out['E_p'],param_out['V_p']=self.e_list,self.V_list
        
        dat_struct={'aim_params':flat_aim(param_out)[0].reshape(1,len(flat_aim(param_out)[0])),
                    'beta':[self.beta],
                    'G_tau':G_tau,
                    'G_l':G_l,
                    'logfile-p1_DC':[self.logfile_p1],
                    'Z':[self.Z],
                    'n_imp':[-1*G_tau[-1]]
                    }

        for key in dat_struct:
                np.savetxt(os.path.join(root,f'{key}.dat'),dat_struct[key])
        
        if not self.self_consistent: self.result_validation(root)

        return
    
    def result_validation(self,root):
        G_iw_recon=self.G_iw.copy()
        Sig_iw_recon=self.G_iw.copy()
        G0_ced=self.G_iw.copy()

        density_from_ed_code=self.G_iw.density()[0][0].real
        G_iw_recon << inverse(inverse(self.G_0_iw) - self.Sigma_iw) #reconstruct from G0_iw
        recon_density = G_iw_recon.density()[0][0].real

        G0_ced << inverse(inverse(self.G_iw)+self.Sigma_iw)

        ed_ced_Z=triqs_interface.quasi_particle_weight(self.Sigma_iw,self.beta)
        Sig_iw_recon << inverse(self.G_0_iw) - inverse(self.G_iw)
        recon_Z=triqs_interface.quasi_particle_weight(Sig_iw_recon,self.beta)

        def _plot():
            seed='err'
            from triqs.plot.mpl_interface import plt,oplot
            plt.figure()
            oplot(self.Sigma_iw,label='ced')
            oplot(Sig_iw_recon,label='rec')
            plt.xlim(-10,10)
            plt.savefig(os.path.join(root,f'{seed}_Sigma.png'))

            plt.figure()
            oplot(self.G_iw,label='ced')
            oplot(G_iw_recon,label='rec')
            plt.xlim(-10,10)
            plt.savefig(os.path.join(root,f'{seed}_G.png'))

            plt.figure()
            oplot(G0_ced,label='ced')
            oplot(self.G_0_iw,label='rec')
            plt.xlim(-10,10)
            plt.savefig(os.path.join(root,f'{seed}_G0.png'))

        if not np.abs(density_from_ed_code-recon_density)<1e-2:
            print(f"{density_from_ed_code-recon_density=}")
            print(f"{self.param}")

            _plot()

            warnings.warn('err: density fail to reconstruct')
            # raise ValueError('err: density fail to reconstruct')
        
        if not np.abs(ed_ced_Z-recon_Z)<1e-2:
            print(f"{ed_ced_Z-recon_Z=}")
            print(f"{self.param}")

            _plot()
            warnings.warn('err: Z fail to reconstruct')
            # raise ValueError('err: Z fail to reconstruct')


def logfilep1():
    file='./ED_RUN/logfile-p1'
    f = open(file, "r")
    lines=f.readlines()
    f.close()
    str_2_look='# double   occ. ='
    DC=np.nan
    for line in lines:
        if str_2_look in line:
            DC=float(line.split()[-1])
            return DC
    if DC is np.nan:
        raise ValueError('DC in ./ED_RUN/logfile-p1 not find')
    return

def format_bath_for_ced_ed(e_list, V_list):
        V_list=np.array(V_list).flatten()
        e_list=np.array(e_list).flatten()

        bath=np.concatenate((V_list,e_list))
        # print("------ BATH FORMAT FOR CED SOLVER IS -------")
        # print(bath)
        return bath

def run_ced_ed_solver(path, omp_threads):
    os.chdir(path)
    ed="dmft_solver "
    output="&> solv_ex"
    args="env OMP_NUM_THREADS=" + str(omp_threads) + " "
    # args_p = " min_all_bath_param=" +str(bath_size) + " " # not need to add this line
    # exe = args + ed + args_p + output#
    exe = args + ed + output
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # print(dir_path)
    proc = subprocess.Popen(exe,
                            shell=True,
                            executable='/usr/bin/bash',
                            stdout=sys.stdout,
                            stderr=sys.stderr)
    proc.wait()
    
    g=read_data_triqs("g1.inp")
    sig=read_data_triqs("sig1.inp")

    os.chdir("../")
    # rm dir!!
    return g,sig

def read_data_triqs(src_file):

    """
    Reads data from ED format and converts to TRIQS GF mesh format

    Parameters:
    argument1 (str)  : GF file

    Returns:
    g (np array)   : GF on TRIQS matsu mesh

    """
    gf=np.loadtxt(src_file).T
    gfr=np.concatenate((np.flip(gf[0]),gf[0]))
    gfi=np.concatenate((np.flip(-gf[1]),gf[1]))
    gf = gfr + 1j*gfi

    return gf

def gf_to_triqs(g,b_v,nomg,label):
    gf=GfImFreq(indices=[0], beta=b_v, n_points = nomg, name=label)
    gf.data[:,0,0] = g
    #get_obj_prop(gf)
    return gf


def save_gf(filename, G):
    data = np.asarray(get_data(G))
    # print(len(data[0]))
    target_n_tau=128
    skip_factor=int((len(data[0])-1)/(target_n_tau))
    np.savetxt(filename, data.T[::skip_factor])   #ES BUG: remove hardcode here

def get_data(G):
    mesh_=[]
    for t in G.mesh:
        mesh_.append(t.value)
    return [mesh_, G.data[:,0,0].real]

def get_data_basis(G, basis):
    mesh_=[]
    if basis == "iw":
        for t in G.mesh:
            mesh_.append(t.value)
        #return [mesh_, G.data[:,0,0].real, G.data[:,0,0].imag]
        return [mesh_, G.data[:,0,0]]
    if basis == "tau":
        for t in G.mesh:
            mesh_.append(t.value)
        return [mesh_, G.data[:,0,0].real]
    if basis == "legendre":
        for t in G.mesh:
            mesh_.append(t.value)
        return [mesh_, G.data[:,0,0].real]

def create_ced_ed_inputs_common(ced_ed_path,
                        beta,
                        U,
                        eps,
                        bath_size,
                        n_iw,
                        mu,
                        ):

    if os.path.exists(ced_ed_path):
        shutil.rmtree(ced_ed_path)

    os.makedirs(ced_ed_path+"/ED")

    f = open('./ED_RUN/PARAMS','w')
    f.write("""1
{eps_up}
{eps_down}
{U}
{n_iw}
{n_iw}
{n_iw}
F
1
F
0.0000000000000000
0.0000000000000000
{mu}
{beta}
0.0
1.0000000000000001E-002
3.4000000000000000E-001
3.2999999999999999E-001
7.0999999999999997E-001
10
0.000000000000000
F
F
F
F
0.0""".format(eps_up=eps,eps_down=eps,n_iw=n_iw,U=U,beta=beta,mu=mu)) # replace with eps
    f.close()



    f = open('./ED_RUN/ED/ed_correl1','w')
    f.write(""" ###############################################
 ### ELECTRONIC CORRELATIONS ON THE IMPURITY ###
 ###############################################
 ########################################
 ###     MASK FOR GREENS FUNCTION    ###
 ### [ <c[up]*C[up]>  <c[up]*c[do]> ] ###
 ### [ <C[do]*C[up]>  <c[do]*C[do]> ] ###
 ########################################
     1   0
     0   2
 ###########################################
 ### MASKS FOR SPIN/DENSITY CORRELATIONS ###
 ###########################################
 F
 F
 F
     1
     2.00000000     # wmax = real freq. range [-wmax,wmax] for Sz
     2.00000000     # wmax = real freq. range [-wmax,wmax] for S+-
     30.0000000     # wmax = real freq. range [-wmax,wmax] for N
 ##################################
 ### MASK FOR Pijk CORRELATIONS ###
 ##################################
 F
             0
 # LIST OF TRIPLETS
 # MASK OF CORRELATIONS
     4.00000000
 ###################################
 ### MASK FOR Pijkl CORRELATIONS ###
 ###################################
 F
             0


     4.00000000
 ##################################################
 ### LIST OF PROJECTION VECTORS IN BASIS |.ud2> ###
 ##################################################
             0
    """)
    f.close()

    f = open('./ED_RUN/ED/ED.in','w')
    f.write("""
PAIRING_IMP_TO_BATH=.false.
track_sectors=.true.
fast_fit=.false.
first_iter_use_edinput=.false.
start_para=.false.
force_nupdn_basis=.true.
force_sz_basis=.false.
force_no_pairing=.true.
lambda_sym_fit=0.p
fit_shift=0.0d0
weight_expo=2
FLAG_MPI_GREENS=1
window_hybrid=0
window_hybrid2=0
window_weight=0.
search_step=0.000001
cutoff_hamilt_param=0.000000010
dist_max=1.d-6
tolerance=1.d-6
FIT_METH=CIVELLI
flag_introduce_only_noise_in_minimization=.false.
flag_introduce_noise_in_minimization=.false.
flag_idelta_two_scales_ed=0
nsec=-1
nsec0=0
which_lanczos=NORMAL
Block_size=0
ncpt_approx=0
cpt_upper_bound=1.00000000
cpt_lagrange=0.00000001
iwindow=1
fmos_iter=1
fmos_mix=0.40000000
fmos=.false.
fmos_fluc=.false.
fmos_hub1=.false.
ncpt_flag_two_step_fit=.false.
Niter_search_max_0=800000
FLAG_ALL_GREEN_FUNC_COMPUTED=.true.
force_para_state=.true.
FLAG_GUP_IS_GDN=.true.
fit_weight_power=1.00000000
fit_nw=100
min_all_bath_param={bath_size}
FLAG_DUMP_INFO_FOR_GAMMA_VERTEX=.false.
Nitermax=1000
Nitergreenmax=500
Neigen=10
# FLAG_FULL_ED_GREEN=.false.
dEmax0=20
Block_size=0
tolerance=1.d-12
diag_bath=.true.
    """.format(bath_size=bath_size))
    f.close()