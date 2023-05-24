import numpy as np
from triqs_hubbardI import Solver as HubbardI_solver
import triqs.gf as triqs_gf
import triqs.operators as triqs_op
from ML_dmft.utility.db_gen import get_data
from ML_dmft.utility.tools import flat_aim,EarlyStopper_DMFT,iw_obj_flip
from ML_dmft.triqs_interface.triqs_interface import hyb
from sklearn.metrics import mean_squared_error
from ML_dmft.utility.mpi_tools import mpi_size
from time import perf_counter
import yaml
import os

def RUN_HubbardI(param,solver_params):
    HI = HI_solver(solver_params['gf_param'])
    HI.solve_DMFT(param=param,dmft_param=solver_params['solve dmft'])
    return HI

def RUN_HI_with_param_only(param,beta):
    pt_params=f"""
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
    dmft_on: false
    error_tol: 0.001
    n_iter_max: 10
    converge_iter_tol: 150
    mixing_parameter: 0.8
"""
    solver_params=yaml.safe_load(pt_params)
    HI = HI_solver(solver_params['gf_param'])
    HI.solve_DMFT(param=param,dmft_param=solver_params['solve dmft'])
    G_iw = HI.G_iw.data[:,:,0][HI.n_iw:2*HI.n_iw]
    return G_iw

def RUN_HI_with_delta(param:dict,beta:float,delta_iw:np.ndarray):
    niw=delta_iw.shape[0]
    pt_params=f"""
gf_param:
    beta: {beta}
    bethe: false
    discrete: true
    discrete_hyb_nk: 8
    indices:
    - 0
    n_iw: {niw}
    n_l: 64
    n_tau: 20480
    target_n_tau: 512
    verbose: 0
"""
    solver_params=yaml.safe_load(pt_params)
    HI = HI_solver(solver_params['gf_param'])
    HI.solve_by_delta_iw(param,delta_iw)
    G_iw = HI.G_iw.data[:,:,0][HI.n_iw:2*HI.n_iw]
    return G_iw

class HI_solver():
    """
    Class for a Hubbard-I solver 
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
        self.Delta_iw = triqs_gf.GfImFreq(indices=self.indices,
                                 beta=self.beta,
                                 n_points=self.n_iw)
        self.G_iw = triqs_gf.GfImFreq(indices=self.indices,
                             beta=self.beta,
                             n_points=self.n_iw)
        self.G_0_iw = triqs_gf.GfImFreq(indices=self.indices,
                             beta=self.beta,
                             n_points=self.n_iw)
        self.Sigma_iw = triqs_gf.GfImFreq(indices=self.indices,
                             beta=self.beta,
                             n_points=self.n_iw)

        # Imaginary time
        self.G_tau_prev = triqs_gf.GfImTime(indices=self.indices,
                                   beta=self.beta,
                                   n_points=self.n_tau)
                                   
        self.G_tau = triqs_gf.GfImTime(indices=self.indices,
                                   beta=self.beta,
                                   n_points=self.n_tau)

        self.G0_tau = triqs_gf.GfImTime(indices=self.indices,
                                   beta=self.beta,
                                   n_points=self.n_tau)

        self.Sigma_tau = triqs_gf.GfImTime(indices=self.indices,
                                   beta=self.beta,
                                   n_points=self.n_tau)

        # Legendre
        self.G_l = triqs_gf.GfLegendre(indices = self.indices, beta = self.beta,
                              n_points = self.n_l)   

        self.Solver = HubbardI_solver(beta = self.beta,
            n_iw=self.n_iw,
            gf_struct = [('up',1), ('down',1) ])
        
    def solve_by_delta_iw(self,param:dict,delta_iw:np.ndarray):
        """
        delta_iw semi positive define.
        """
        self.U = param['U']
        self.eps = param['eps']
        self.mu = -self.eps-self.U/2
        self.Delta_iw.data[:,:,0]=iw_obj_flip(delta_iw)
        self.G_0_iw << triqs_gf.inverse(triqs_gf.iOmega_n + self.mu -self.eps - self.Delta_iw)
        self.h1_solve(self.G_0_iw)

    def solve_DMFT(self,param,dmft_param):
        self.eps = param["eps"]
        self.U = param["U"]

        if self.bethe:
            self.W = param["W"]
        elif self.discrete:
            self.e_list = param["E_p"]
            self.V_list = param["V_p"]
        else :
            raise ValueError('bethe or discrete not defined')
        
        if self.verbose >= 1:
            if np.abs(-self.eps*2-self.U) > 1e-5:
                print('not doing half-filled!!')

        # self.mu=-self.eps-self.U # self.eps general -eps-U/2-U/2
        self.mu = -self.eps-self.U/2


        #dmft params
        self_consistent = dmft_param['dmft_on']
        tol = dmft_param['error_tol']
        n_iter_max = dmft_param['n_iter_max']
        dmft_tolerance = dmft_param['converge_iter_tol']
        alpha=dmft_param['mixing_parameter']

        ES=EarlyStopper_DMFT(dmft_tolerance,0)

        last_time=perf_counter()

        if self_consistent:
            if self.verbose >=2: print('Running dmft')
        else:
            if self.verbose >=2: print('one iteration only')


        if self.discrete:
            self.Delta_iw << hyb(param["E_p"], param["V_p"]) 
        if self.bethe:
            self.Delta_iw << ((self.W**2)/4.)*triqs_gf.SemiCircular(self.W) 
            
        for iter_num in range(n_iter_max):

            self.G_0_iw << triqs_gf.inverse(triqs_gf.iOmega_n + self.mu -self.eps - self.Delta_iw)

            self.G_tau_prev <<  triqs_gf.Fourier(self.G_iw)
            self.G0_tau <<  triqs_gf.Fourier(self.G_0_iw)

            # self.Sigma_iw << self.ipt_self_enerngy() ## shift by U/2
            # self.Sigma_iw << self.h1_solve(self.G_0_iw)
            self.h1_solve(self.G_0_iw)
            # self.G_iw << triqs_gf.inverse(triqs_gf.inverse(self.G_0_iw) - self.Sigma_iw) #dyson find G_iw

            # self.Sigma_iw << inverse(self.G_0_iw)-inverse(self.G_iw) #
            self.Z=1 / (1 - (self.Sigma_iw(0)[0,0].imag * self.beta / np.pi)) 
            self.G_tau << triqs_gf.Fourier(self.G_iw)
            self.G_l << triqs_gf.MatsubaraToLegendre(self.G_iw)


        # checking if exist
            
            if not self_consistent:
                if self.verbose >= 1:
                    print("running 1 iteration only")
                break

            # measaure err
            err=max(np.abs(self.G_tau_prev.data[:,0,0].real-self.G_tau.data[:,0,0].real))

            if err < tol and iter_num > 5:
                if self.verbose >= 1: print("err max {}".format(err))
                print(f"{iter_num=}, {err=}, err tolerence={tol}")
                break

            if self.verbose >= 2: 
                print('err {:.10f} {:d} {:5f}'.format(err,iter_num,perf_counter()-last_time))
                last_time=perf_counter()


            if  iter_num > dmft_tolerance and ES.early_stop(err):
                if self.verbose >= 1: print("U={} Fail to converge {}".format(self.U,iter_num))
                if self.verbose >= 1: print(f"{ES.min_err=} {err=}")
                if self.verbose >= 1: print("err max {}".format(err))
                if self.verbose >= 1: print('err mse',mean_squared_error(self.G_tau_prev.data[:,0,0].real,self.G_tau.data[:,0,0].real))
                break
        
        # updating hyb
            if self.bethe:
                Delta_new_iw = self.Delta_iw.copy()
                Delta_new_iw << self.W**2*self.G_iw/4 
                self.Delta_iw << alpha * Delta_new_iw + (1-alpha) * self.Delta_iw
            else:
                raise ValueError('define the lattice type properly')

            if iter_num == n_iter_max-1:
                print(f"{ES.min_err=} {err=}")
                print('err',mean_squared_error(self.G_tau_prev.data[:,0,0].real,self.G_tau.data[:,0,0].real))
                print("Solution not converged! after %d"%iter_num)


    def h1_solve(self,G0:triqs_gf.GfImFreq):
        for _name, g0 in self.Solver.G0_iw:
            # g0 << inverse(iOmega_n + self.mu - self.Delta_iw)
            # g0 << triqs_gf.inverse(triqs_gf.iOmega_n + self.mu -self.eps - self.Delta_iw)
            g0 << G0
        self.Solver.solve(h_int = self.U * triqs_op.n('up',0) * triqs_op.n('down',0),
                calc_gw = True)
        # G_iw,Sigma_iw = self.G_iw.copy(),self.G_iw.copy()
        self.G_iw << (self.Solver.G_iw['up']+self.Solver.G_iw['down'])/2.0
        self.Sigma_iw << (self.Solver.Sigma_iw['up']+self.Solver.Sigma_iw['down'])/2.0
        return
    
    def extract_data(self):
        self.G_l << triqs_gf.MatsubaraToLegendre(self.G_iw)

        data_tau = np.asarray(get_data(self.G_tau, "tau"))
        idxs = np.linspace(0, len(data_tau[0]) - 1, self.target_n_tau).astype(int)
        new_tau=np.zeros((1,len(idxs)))
        for i,idx in enumerate(idxs):
            new_tau[0][i]=data_tau[1][idx]

        data_leg = np.asarray(get_data(self.G_l, "legendre"))
        ret_leg = data_leg[1][::1, np.newaxis].T,
        return new_tau, ret_leg

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

        dat_struct={'aim_params':flat_aim(param)[0],
                'beta':[self.beta],
                'G_tau':G_tau,
                'G_l':G_l,
                'Z':[self.Z],
                'n_imp':[-1*G_tau[-1]]
                }
        for key in dat_struct:
            np.savetxt(os.path.join(root,f'{key}.dat'),dat_struct[key])

        return

