# -*- coding: utf-8 -*-
"""
ML DMFT
===============================
# Author: Even Evan Sheridan 2021
# Author: Zelong Zhao 2022
"""


import numpy as np
import os
from triqs.gf import *
import triqs.gf as triqs_gf
import ML_dmft.numpy_GF.GF as GF
import ML_dmft.numpy_GF.common as PYDMFT_GF
from ML_dmft.utility.db_gen import get_data
from ML_dmft.utility.tools import flat_aim,EarlyStopper_DMFT,iw_obj_flip
from ML_dmft.triqs_interface.triqs_interface import hyb
from sklearn.metrics import mean_squared_error
from triqs.lattice import BravaisLattice, BrillouinZone
from ML_dmft.utility.mpi_tools import mpi_size
from ML_dmft.triqs_interface.hysteresis.triqs_tails import self_energy_tails
from time import perf_counter
from numpy import cos
from copy import deepcopy
import yaml
from ML_dmft.utility.dmft_performance_checker import Monitor_Dmft_Performance


def threads_IPT(param,solver_params):
        IPT = PT_solver(solver_params['gf_param'])
        IPT.solve_DMFT(param=param,dmft_param=solver_params['solve dmft'])
        out_data_dict=IPT.saveGF2mydataformat(param)
        return out_data_dict

def RUN_IPT(param,solver_params):
        IPT = PT_solver(solver_params['gf_param'])
        IPT.solve_DMFT(param=param,dmft_param=solver_params['solve dmft'])
        return IPT

def RUN_IPT_with_param_only(param:dict,beta:float):
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
    hatree_shift_back : True
    hartree_shift_err_tol : 0.5
    dmft_on: false
    error_tol: 0.001
    n_iter_max: 10
    converge_iter_tol: 150
    mixing_parameter: 0.8
"""
    solver_params=yaml.safe_load(pt_params)
    IPT = PT_solver(solver_params['gf_param'])
    IPT.solve_DMFT(param=param,dmft_param=solver_params['solve dmft'])
    G_iw = IPT.G_iw.data[:,:,0][IPT.n_iw:2*IPT.n_iw]
    return G_iw

def RUN_IPT_with_delta(param:dict,beta:float,delta_iw:np.ndarray):
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
    IPT = PT_solver(solver_params['gf_param'])
    IPT.solve_by_delta_iw(param,delta_iw)
    G_iw = IPT.G_iw.data[:,:,0][IPT.n_iw:2*IPT.n_iw]
    return G_iw


class PT_solver():
    """
    Class for a perturbation theory solver 
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
        self.G_iw = GfImFreq(indices=self.indices,
                             beta=self.beta,
                             n_points=self.n_iw)
        self.G_0_iw = GfImFreq(indices=self.indices,
                             beta=self.beta,
                             n_points=self.n_iw)
        self.Sigma_iw = GfImFreq(indices=self.indices,
                             beta=self.beta,
                             n_points=self.n_iw)

        # Imaginary time
        self.G_tau_prev = GfImTime(indices=self.indices,
                                   beta=self.beta,
                                   n_points=self.n_tau)
                                   
        self.G_tau = GfImTime(indices=self.indices,
                                   beta=self.beta,
                                   n_points=self.n_tau)

        self.G0_tau = GfImTime(indices=self.indices,
                                   beta=self.beta,
                                   n_points=self.n_tau)

        self.Sigma_tau = GfImTime(indices=self.indices,
                                   beta=self.beta,
                                   n_points=self.n_tau)

        
        self.imtime_target = np.linspace(0, self.beta, 
                                         self.target_n_tau, endpoint=True)

        # Legendre
        self.G_l = GfLegendre(indices = self.indices, beta = self.beta,
                              n_points = self.n_l)        
        self.legendre_target =  np.linspace(1, self.n_l, self.n_l, endpoint = True)

        # define k space
        if self.discrete:
            BL = BravaisLattice([(1, 0, 0), (0, 1, 0)]) # Two unit vectors in R3
            BZ = BrillouinZone(BL) 
            self.n_k=solver_params['discrete_hyb_nk']
            self.kmesh = MeshBrillouinZone(BZ, n_k=self.n_k)
            if self.verbose >=2:
                print('2d square lattice k points',self.n_k**2)

    def solve_by_delta_iw(self,param:dict,delta_iw:np.ndarray):
        """
        delta_iw semi positive define.
        """
        self.U = param['U']
        self.eps = param['eps']
        self.W = param['W']
        self.mu = -self.eps-self.U #shifted

        self.Delta_iw.data[:,:,0]=iw_obj_flip(delta_iw)
        # self.Delta_iw << ((self.W**2)/4.)*SemiCircular(self.W) 
        self.G_0_iw << triqs_gf.inverse(triqs_gf.iOmega_n + self.mu -self.eps - self.Delta_iw)
        self.G0_tau <<  Fourier(self.G_0_iw)

        self.Sigma_iw << self.ipt_self_enerngy_tails() ## shift by U/2
        # self.Sigma_iw << self.ipt_self_enerngy() ## shift by U/2
        self.G_iw << inverse(inverse(self.G_0_iw) - self.Sigma_iw) #dyson find G_iw
        
        # print(f"{self.Delta_iw.data[:,:,0][self.n_iw:self.n_iw+5].T=}")
        # print(f"{self.Sigma_iw.data[:,:,0][self.n_iw:self.n_iw+5].T=}")
        # print(f"{self.G_iw.data[:,:,0][self.n_iw:self.n_iw+5].T=}")

        return self.G_iw

    def epsi(self,k):
        self.t=self.eps-self.mu
        self.t=self.eps # for half filling eps=-U/2. mu=U/2
        return -2 *  self.t * (cos(k[0]) + cos(k[1]))

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

        self.mu=-self.eps-self.U # self.eps general -eps-U/2-U/2
        self.mu_unshifted = -self.eps-self.U/2

        #ipt_param
        self.PT_shifted_back=dmft_param['hatree_shift_back']
        self.hartree_shift_err_tol=dmft_param['hartree_shift_err_tol']

        #dmft params
        self_consistent = dmft_param['dmft_on']
        tol = dmft_param['error_tol']
        n_iter_max = dmft_param['n_iter_max']
        dmft_tolerance = dmft_param['converge_iter_tol']
        alpha=dmft_param['mixing_parameter']

        plot_converge=False

        ES=EarlyStopper_DMFT(dmft_tolerance,0)

        last_time=perf_counter()

        if self_consistent:
            if self.verbose >=2: print('Running dmft')
            if self.verbose >=4: plot_converge=True
            if plot_converge: DMFT_Monitor=Monitor_Dmft_Performance(f"U_{int(param['U'])}")
        else:
            if self.verbose >=2: print('one iteration only')


        if self.discrete:
            self.Delta_iw << hyb(param["E_p"], param["V_p"]) 
        if self.bethe:
            self.Delta_iw << ((self.W**2)/4.)*SemiCircular(self.W) 
            
        for iter_num in range(n_iter_max):
            self.G_0_iw << inverse(iOmega_n +self.mu-self.eps - self.Delta_iw)
            self.G_0_iw_unshifited =  self.G_0_iw.copy()
            self.G_0_iw_unshifited << inverse(iOmega_n + self.mu_unshifted -self.eps - self.Delta_iw)

            self.G_tau_prev <<  Fourier(self.G_iw)
            self.G0_tau <<  Fourier(self.G_0_iw)

            self.Sigma_iw << self.ipt_self_enerngy() ## shift by U/2
            self.G_iw << inverse(inverse(self.G_0_iw) - self.Sigma_iw) #dyson find G_iw
            # self.Sigma_iw << inverse(self.G_0_iw)-inverse(self.G_iw) #
            self.Z=1 / (1 - (self.Sigma_iw(0)[0,0].imag * self.beta / np.pi)) 
            self.G_tau << Fourier(self.G_iw)
            self.G_l << MatsubaraToLegendre(self.G_iw)


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

            if plot_converge:
                DMFT_Monitor.init_oneframe()
                DMFT_Monitor.plot_oneframe(GF.matsubara_freq(self.beta,32),self.Delta_iw.data[:,:,0][self.n_iw:],'delta')
                DMFT_Monitor.plot_oneframe(GF.matsubara_freq(self.beta,32),self.G_iw.data[:,:,0][self.n_iw:],'IPT')
                DMFT_Monitor.finish_oneframe()

        # updating hyb
            if self.discrete:
                Delta_new_iw = self.Delta_iw.copy()
                Delta_new_iw << self.update_hyb()
                self.Delta_iw << alpha * Delta_new_iw + (1-alpha) * self.Delta_iw

            elif self.bethe:
                Delta_new_iw = self.Delta_iw.copy()
                Delta_new_iw << self.W**2*self.G_iw/4 
                self.Delta_iw << alpha * Delta_new_iw + (1-alpha) * self.Delta_iw
            else:
                raise ValueError('define the lattice type properly')

            if iter_num == n_iter_max-1:
                print(f"{ES.min_err=} {err=}")
                print('err',mean_squared_error(self.G_tau_prev.data[:,0,0].real,self.G_tau.data[:,0,0].real))
                print("Solution not converged! after %d"%iter_num)


        if plot_converge: DMFT_Monitor.finish_allframe()
        #hartree shift
        self.Z=1 / (1 - (self.Sigma_iw(0)[0,0].imag * self.beta / np.pi)) 
        if self.PT_shifted_back: 
            if self.verbose >= 2:
                print('hartree shift back')
            self.shifted_back()

    
    def solve_DMFT_back(self,param,dmft_param):
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

        self.mu=-self.eps-self.U # self.eps general -eps-U/2-U/2
        self.mu_unshifted = -self.eps-self.U/2

        #ipt_param
        self.PT_shifted_back=dmft_param['hatree_shift_back']
        self.hartree_shift_err_tol=dmft_param['hartree_shift_err_tol']

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
            self.Delta_iw << ((self.W**2)/4.)*SemiCircular(self.W) 
            
        for iter_num in range(n_iter_max):

            self.G_0_iw << inverse(iOmega_n +self.mu-self.eps - self.Delta_iw)
            self.G_0_iw_unshifited =  self.G_0_iw.copy()
            self.G_0_iw_unshifited << inverse(iOmega_n + self.mu_unshifted -self.eps - self.Delta_iw)

            self.G_tau_prev <<  Fourier(self.G_iw)
            self.G0_tau <<  Fourier(self.G_0_iw)

            self.Sigma_iw << self.ipt_self_enerngy() ## shift by U/2
            self.G_iw << inverse(inverse(self.G_0_iw) - self.Sigma_iw) #dyson find G_iw
            # self.Sigma_iw << inverse(self.G_0_iw)-inverse(self.G_iw) #
            self.Z=1 / (1 - (self.Sigma_iw(0)[0,0].imag * self.beta / np.pi)) 
            self.G_tau << Fourier(self.G_iw)
            self.G_l << MatsubaraToLegendre(self.G_iw)


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
            if self.discrete:
                Delta_new_iw = self.Delta_iw.copy()
                Delta_new_iw << self.update_hyb()
                self.Delta_iw << alpha * Delta_new_iw + (1-alpha) * self.Delta_iw

            elif self.bethe:
                Delta_new_iw = self.Delta_iw.copy()
                Delta_new_iw << self.W**2*self.G_iw/4 
                self.Delta_iw << alpha * Delta_new_iw + (1-alpha) * self.Delta_iw
            else:
                raise ValueError('define the lattice type properly')

            if iter_num == n_iter_max-1:
                print(f"{ES.min_err=} {err=}")
                print('err',mean_squared_error(self.G_tau_prev.data[:,0,0].real,self.G_tau.data[:,0,0].real))
                print("Solution not converged! after %d"%iter_num)

        #hartree shift
        self.Z=1 / (1 - (self.Sigma_iw(0)[0,0].imag * self.beta / np.pi)) 
        if self.PT_shifted_back: 
            if self.verbose >= 2:
                print('hartree shift back')
            self.shifted_back()
                        

    def Bethe_DMFT_Giw_inputs(self,param:dict,dmft_param:dict,g_iw:np.ndarray):
        """
        param = dict(U,eps,W)
        g_iw = np.ndarray semi-positive defined.
        """
        self.eps = param["eps"]
        self.U = param["U"]
        self.W = param["W"]

        g_iw,G_iw_triqs = GF.iw_obj_flip(g_iw),self.G_iw.copy()
        G_iw_triqs.data[:,:,0] = g_iw

        
        if self.verbose >= 1:
            if np.abs(-self.eps*2-self.U) > 1e-5:
                print('not doing half-filled!!')

        self.mu=-self.eps-self.U # self.eps general -eps-U/2-U/2
        self.mu_unshifted = -self.eps-self.U/2

        #ipt_param
        self.PT_shifted_back=dmft_param['hatree_shift_back']
        self.hartree_shift_err_tol=dmft_param['hartree_shift_err_tol']

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

        # self.Delta_iw << ((self.W**2)/4.)*SemiCircular(self.W) 
        self.Delta_iw << ((self.W**2)/4.)*G_iw_triqs
            
        for iter_num in range(n_iter_max):

            self.G_0_iw << inverse(iOmega_n +self.mu-self.eps - self.Delta_iw)
            self.G_0_iw_unshifited =  self.G_0_iw.copy()
            self.G_0_iw_unshifited << inverse(iOmega_n + self.mu_unshifted -self.eps - self.Delta_iw)

            self.G_tau_prev <<  Fourier(self.G_iw)
            self.G0_tau <<  Fourier(self.G_0_iw)

            self.Sigma_iw << self.ipt_self_enerngy_tails() ## shift by U/2
            self.G_iw << inverse(inverse(self.G_0_iw) - self.Sigma_iw) #dyson find G_iw
            self.G_iw.data.real = 0

            # self.Sigma_iw << inverse(self.G_0_iw)-inverse(self.G_iw) #
            self.Z=1 / (1 - (self.Sigma_iw(0)[0,0].imag * self.beta / np.pi)) 
            self.G_tau << Fourier(self.G_iw)
            self.G_l << MatsubaraToLegendre(self.G_iw)


        # checking if exist
            if not self_consistent:
                if self.verbose >= 1:
                    print("running 1 iteration only")
                break

            Delta_new_iw = self.Delta_iw.copy()
            Delta_new_iw << self.W**2*self.G_iw/4 
            self.Delta_iw << alpha * Delta_new_iw + (1-alpha) * self.Delta_iw

            # measaure err
            err=max(np.abs(self.Delta_iw.data[:,:,0].imag-Delta_new_iw.data[:,:,0].imag))
            # err=max(np.abs(self.G_tau_prev.data[:,0,0].real-self.G_tau.data[:,0,0].real))

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
        
            Delta_new_iw = self.Delta_iw.copy()
            Delta_new_iw << self.W**2*self.G_iw/4 
            self.Delta_iw << alpha * Delta_new_iw + (1-alpha) * self.Delta_iw
        

            if iter_num == n_iter_max-1:
                print(f"{ES.min_err=} {err=}")
                print('err',mean_squared_error(self.G_tau_prev.data[:,0,0].real,self.G_tau.data[:,0,0].real))
                print("Solution not converged! after %d"%iter_num)

        #hartree shift
        self.Z=1 / (1 - (self.Sigma_iw(0)[0,0].imag * self.beta / np.pi)) 
        if self.PT_shifted_back: 
            if self.verbose >= 2:
                print('hartree shift back')
            self.shifted_back()

    def ipt_self_enerngy_tails(self):
        self.Sigma_tau << (self.U**2) * self.G0_tau * self.G0_tau * self.G0_tau
        self.Sigma_iw << self_energy_tails(self.Sigma_tau,
                                        U=self.U,beta=self.beta,n_iw=self.n_iw)
        return self.Sigma_iw

    def get_output_giw_z(self):
        print(f"{self.U} {self.beta}")

        return self.G_iw.data[:,:,0][self.n_iw:],self.Z

    def update_hyb(self):
        Delta_new_iw = GfImFreq(indices=self.indices,
                beta=self.beta,
                n_points=self.n_iw)
        if self.verbose >=2:
            print('updating discrete hyb with mu %.2f'%self.mu)
        for k in self.kmesh:
            Delta_new_iw << Delta_new_iw + inverse(iOmega_n+self.mu-self.epsi(k)-self.Sigma_iw)
        Delta_new_iw << Delta_new_iw/(self.n_k**2)
        Delta_new_iw << iOmega_n + self.mu - self.Sigma_iw - inverse(Delta_new_iw)
        return Delta_new_iw

    def solv_AIM(self):
        """
        PT solver for the AIM
        """
        # self.Sigma_iw << calc_SE(self.G_tau_prev, self.U) # ES
        self.Sigma_iw << self.ipt_self_enerngy() ## ZZ
        self.Z=1 / (1 - (self.Sigma_iw(0)[0,0].imag * self.beta / np.pi)) 
        self.G_iw << inverse(inverse(self.G_0_iw) - self.Sigma_iw) #dyson find G_iw
        self.G_tau << Fourier(self.G_iw)
        return self.G_tau

    def shifted_back(self):
        Z_prev=1 / (1 - (self.Sigma_iw(0)[0,0].imag * self.beta / np.pi))

        self.G_0_iw << self.G_0_iw_unshifited
        self.Sigma_iw << inverse(self.G_0_iw) - inverse(self.G_iw) 

        # self.Sigma_iw.data[:,0,0].real=self.Sigma_iw.data[:,0,0].real+self.U/2

        self.Z=1 / (1 - (self.Sigma_iw(0)[0,0].imag * self.beta / np.pi)) 

        if self.verbose >= 2:
            print(f"{self.mu-self.eps=} {self.U=}")
            print(self.G_iw.density()[0][0].real)
        if np.abs(Z_prev-self.Z) > 1e-1:
            print("Z {}".format(self.Z))
            print('Z shifted!')
            print(Z_prev-self.Z)
            raise ValueError('Z after re shift')

    def ipt_self_enerngy(self):
        self.Sigma_tau << (self.U**2) * self.G0_tau * self.G0_tau * self.G0_tau
        return Fourier(self.Sigma_tau) #+ self.U/2. # have to do this
        
    def extract_data(self):
        self.G_l << MatsubaraToLegendre(self.G_iw)

        data_tau = np.asarray(get_data(self.G_tau, "tau"))
        idxs = np.linspace(0, len(data_tau[0]) - 1, self.target_n_tau).astype(int)
        new_tau=np.zeros((1,len(idxs)))
        for i,idx in enumerate(idxs):
            new_tau[0][i]=data_tau[1][idx]

        data_leg = np.asarray(get_data(self.G_l, "legendre"))
        ret_leg = data_leg[1][::1, np.newaxis].T,
        return new_tau, ret_leg
    
    def extract_data_G0(self):
        G_0_tau=self.G_tau.copy()
        G_0_tau << Fourier(self.G_0_iw)

        G_0_l = self.G_l.copy()
        G_0_l << MatsubaraToLegendre(self.G_0_iw)
        
        data_tau = np.asarray(get_data(G_0_tau, "tau"))

        idxs = np.linspace(0, len(data_tau[0]) - 1, self.target_n_tau).astype(int)
        new_tau=np.zeros((1,len(idxs)))
        for i,idx in enumerate(idxs):
            new_tau[0][i]=data_tau[1][idx]

        data_leg = np.asarray(get_data(G_0_l, "legendre"))
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

        if self.bethe:
            del param_out['E_p']
            del param_out['V_p']
            
        elif self.discrete:
            del param_out['W']
        else :
            raise ValueError('bethe or discrete not defined')

        dat_struct={'AIM params':param_out,
            'data info':data_info,
            'G_tau':G_tau,
            'G_l':G_l,
            'Sigma_iw':self.Sigma_iw.data[:,:,0].T,
            'G0_iw':self.G_0_iw.data[:,:,0].T,
            'G_iw':self.G_iw.data[:,:,0].T,
            'Z':self.Z,
            'bethe':self.bethe
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

    def solv_AIM_iteration(self):
        """
        PT solver for the AIM
        """
        self.pt_order="3"
        self_consistent = True
        tol, n_iter_max = 1e-6, 100
        for iter_num in range(n_iter_max):
            
            if self.pt_order == "1":
                self.Sigma_iw << (first_order_Sigma(self.G_tau_prev, self.U))
                              
            if self.pt_order == "2":
                self.Sigma_iw << (first_order_Sigma(self.G_tau_prev, self.U) \
                                  + second_order_Sigma(self.G_tau_prev, self.U, only_skeleton=self_consistent))
                
            if self.pt_order == "3": 
                self.Sigma_iw << (first_order_Sigma(self.G_tau_prev, self.U) \
                                  + second_order_Sigma(self.G_tau_prev, self.U, only_skeleton=self_consistent) \
                                  + third_order_Sigma(self.G_tau_prev, self.U, self.indices, 
                                                      self.n_iw, only_skeleton=self_consistent))
            
            self.G_iw << inverse(inverse(self.G_0_iw) - self.Sigma_iw)
            self.G_tau << Fourier(self.G_iw)
            
            if np.allclose(self.G_tau_prev.data, self.G_tau.data, atol=tol) or not self_consistent:
                print("Converged in iteration {}".format(iter_num))
                return self.G_tau
            else:
                self.G_tau_prev << 0.8 * self.G_tau + 0.2 * self.G_tau_prev
                # print("Solution not converged!")
        return self.G_tau
    
def reverse_tau(G_tau, statistic="Fermion"):
    sign = -1 if statistic == "Fermion" else 1
    G_minus_tau = G_tau.copy()
    G_minus_tau.data[:,0,0] = sign * np.flipud(G_tau.data[:,0,0])
    # ES BUG: TODO add tail with TRIQS V3
    # for m in range(G_tau.tail.order_min, G_tau.tail.order_max + 1)    
    #     G_minus_tau.tail[m] = (-1)**m * G_tau.tail[m]
    return G_minus_tau

def trapez(X, dtau):
    if len(X) < 2: return 0
    I = dtau * np.sum(X[1:-1])
    I += 0.5 * dtau * (X[0] + X[-1])
    return I
        
def integration(X_tau):
    dtau = X_tau.mesh.beta / (len(X_tau.data) - 1)
    return trapez(X_tau.data[:,0,0], dtau)


def convolution(X_tau, Y_tau, n_iw, indices,  statistic="Fermion"):
    X_iw = GfImFreq(indices=indices, beta=X_tau.mesh.beta, n_points=n_iw, statistic=statistic)
    Y_iw = GfImFreq(indices=indices, beta=X_tau.mesh.beta, n_points=n_iw, statistic=statistic)
    X_iw << Fourier(X_tau if X_tau.mesh.statistic == statistic else change_statistic(X_tau))
    Y_iw << Fourier(Y_tau if Y_tau.mesh.statistic == statistic else change_statistic(Y_tau))
    Z_tau = GfImTime(indices=indices, beta=X_tau.mesh.beta, n_points=(len(X_tau.data)),
                     statistic=statistic)
    Z_tau << Fourier(X_iw * Y_iw)
    return Z_tau if X_tau.mesh.statistic == statistic else change_statistic(Z_tau)

def first_order_Sigma(G_tau, U):
    n = G_tau.data[0,0,0].real + 1
    return U * (n - 0.5)

def second_order_Sigma(G_tau, U, only_skeleton=False):
    Sigma_tau = G_tau.copy()
    G_minus_tau = reverse_tau(G_tau)
    Sigma_tau << -U**2 * G_tau * G_tau * G_minus_tau
    # non-skeleton contributions
    Hartree = U/2
    if not only_skeleton:
        Hartree = U * first_order_Sigma(G_tau, U) * integration(G_minus_tau * G_tau)
    return Fourier(Sigma_tau) + Hartree

def third_order_Sigma(G_tau, U, indices, n_iw, only_skeleton=False):
    Sigma_tau = G_tau.copy()
    G_minus_tau = reverse_tau(G_tau)
    # skeleton contributions 3a and 3b
    Sigma = U**3 * G_tau * convolution(G_tau * G_minus_tau, G_tau * G_minus_tau,  n_iw, indices, "Boson")
    Sigma +=  U**3 * G_minus_tau * convolution(G_tau * G_tau, G_tau * G_tau,  n_iw, indices, "Boson")
    # non-skeleton contributions
    Hartree = 0
    if not only_skeleton:
        tadpole = first_order_Sigma(G_tau, U)
        # Diagrams 3c and 3e
        X_tau = convolution(G_tau, G_tau, n_iw, indices)
        Sigma += -tadpole * U**2 * G_tau * G_minus_tau * X_tau * 2
        # Diagram 3d
        Sigma += -tadpole * U**2 * G_tau * G_tau * reverse_tau(X_tau)
        # Hartree diagrams 3a, 3b, 3c
        Hartree += tadpole * U**2 * integration(G_minus_tau * G_tau)**2
        Hartree += tadpole**2 * U * integration(G_minus_tau * X_tau)
        X_tau = convolution(G_tau * G_tau * G_minus_tau, G_tau, n_iw, indices)
        Hartree += -U**3 * integration(G_minus_tau * X_tau)
    Sigma_tau << Sigma
    return Fourier(Sigma_tau) + Hartree
    
def intg(xt):
    dtau = xt.mesh.beta/(len(xt.data) - 1)
    arg = dtau*np.sum(xt.data[:,0,0][1:-1])
    arg = arg+0.5*dtau*(xt.data[:,0,0][0]+xt.data[:,0,0][-1])    
    return arg

def calc_SE(G_tau, U):
    ft = U*(G_tau.data[0,0,0].real+0.5)
    Grev = G_tau.copy()
    Grev.data[:,0,0] = -np.flipud(G_tau.data[:,0,0])
    Sigma_tau = G_tau.copy()
    Sigma_tau << -U**2*G_tau*G_tau*Grev
    ha = U*ft*intg(Grev*G_tau)
    ret_val = ft+Fourier(Sigma_tau)+ha
    return ret_val
        
