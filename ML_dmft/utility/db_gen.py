import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from triqs.gf import *
from triqs.dos import *
from ML_dmft.utility.tools import *
from copy import deepcopy
from ML_dmft.numpy_GF.fit_hyb import cal_err_semi_circular_DOS_symmetry


class db_AIM():
    """
    Class for the database object
    """

    def __init__(self, args):
        """
        Initialise the database. 

        Parameters
        ----------

        beta: scalar
        Inverse temparature

        U: list of length 2 of reals
        Range betwen U_min and U_max

        eps: list of length 2 of reals
        Range betwen eps_min and eps_max

        D: list of length 2 of reals
        Range betwen D_min and D_max (bandwidth)

        V: list of length 2 of reals
        Range betwen D_min and D_max (bath parameters)

        N: integer list 
        List of the number of bath parameters used 

        n_entries: integer
        number of entries per core in the database

        filename: string 
        name of the csv file for the database
        
        """
        self.U=args['U_']        
        self.eps = args["eps_"]
        self.W = args["D_"] #half bandwidth
        self.V = args["V_"]
        self.e = args["e_"]
        self.N = int(args["N_"])
        self.n_entries = args['n_samples']
        self.half_filled = args["half_filled"]
        self.bethe = args["bethe"]
        self.mott_num = args['mott_num']
        self.U_0_shift = args["U_0_shift"]
        self.beta = args['beta']

        self.fit_err_tol = args['fit_err_tol']
        self.delta_from_zero = args['delta_from_zero']
        self.fit_method = args['fit_method']
        self.fit_max_iter = args['fit_max_iter']
        self.init_e_bound = args['init_e_bound']
        self.init_V_bound = args['init_V_bound']
        self.max_fit_err_tol = args['max_fit_err_tol']
        self.err_tol_fix = args['err_tol_fix']
        self.minimizer_maxiter = args['minimizer_maxiter']

        self.data_entries = []

        
    def aim_p(self):
        """
        Generate a set of random parameters for the database
        """
        
        U = make_me_rand(self.U) 
        if self.half_filled:
            eps = -U/2
        else:
            eps = make_me_rand(self.eps) 

        W = make_me_rand(self.W) 
        N = self.N

        V_arr=make_array_rand(N,self.V)
        e_arr=make_array_rand(N,self.e)
        
        params = [U, eps, W, N] + e_arr.tolist() + V_arr.tolist()

        return params

    def half_filled_aim_p(self):
        """
        Generate a set of random parameters for the database
        """
        
        U = make_me_rand(self.U) 
        if self.half_filled:
            eps = -U/2
        else:
            eps = make_me_rand(self.eps) 

        W = make_me_rand(self.W) 
        N = self.N

        # assert int(N%2)==0,'Number of bath sites must be even number'

        if self.bethe:

            fit_semi_circular_dict=dict(num_imp=N,
                            beta=self.beta,
                            err_tol=self.fit_err_tol,
                            max_fitting_err=self.max_fit_err_tol,
                            delta_from_zero=self.delta_from_zero,
                            V_bound = self.V,
                            E_bound = self.e,
                            V_bound_init = self.init_V_bound,
                            E_bound_init = self.init_e_bound,
                            minimizer_maxiter=self.minimizer_maxiter,
                            n_iw=64,
                            W=W, 
                            omega_c=32,
                            err_tol_fix=self.err_tol_fix,
                            method=self.fit_method,
                            max_iter=self.fit_max_iter,
                            disp=True,
                            )
            _,_,_,e_list,V_list=cal_err_semi_circular_DOS_symmetry(**fit_semi_circular_dict)
            params = [U, eps, W, N] + e_list.tolist() + V_list.tolist()
        
        else:

            V_arr=make_array_rand(int(N/2),self.V)
            V_arr=np.concatenate((V_arr,V_arr*-1),axis=None)

            e_arr=make_array_rand(int(N/2),self.e)
            e_arr=np.concatenate((e_arr,e_arr*-1),axis=None)
        
            params = [U, eps, W, N] + e_arr.tolist() + V_arr.tolist()

        return params

            
    def create_db(self):
        """
        create the database 
        """
        if self.half_filled:
            for _ in range(self.n_entries):
                self.data_entries.append(self.half_filled_aim_p())          
        else: 
            for _ in range(self.n_entries):
                self.data_entries.append(self.aim_p())
    
    def create_mott_db(self):

        if self.n_entries < self.mott_num:
            raise ValueError('No enough point to create Mott database')
        if self.n_entries%self.mott_num !=0:
            raise ValueError('n_samples should be multiple of Mott_num')

        for idx in range(self.n_entries):
            if idx%self.mott_num==0:
                if self.bethe:
                    param=self.half_filled_aim_p()
                else:
                    param=self.aim_p()
                U0=np.random.random()
                if self.U_0_shift:
                    U_list=np.linspace(self.U[0]+U0,self.U[1]-U0,self.mott_num)
                else:
                    U_list=np.linspace(self.U[0],self.U[1],self.mott_num)

                for U in U_list:
                    param_out=deepcopy(param)
                    param_out[0]=U
                    eps = -U/2
                    if self.half_filled:
                        param_out[1]=eps

                    self.data_entries.append(param_out)

        
def make_me_rand(arg):
    return np.random.uniform(*arg)

def make_array_rand(N,In):
    left=In[0]
    right=In[1]
    return np.random.uniform(left,right,N)

def get_V_arr(V_inp, N, W):        
    V_ran = np.random.random(N)
    V_arr = (V_inp[1] - V_inp[0]) * V_ran + V_inp[0]
    norm = np.sum(v**2 for v in V_arr)
    scale_V = np.sqrt(2*W/(norm))
    V_arr = V_arr*scale_V
    return V_arr 

def get_E_arr(N, V_arr, W):
    e_arr = np.random.random(N)
    e_arr = np.sort(e_arr)
    arim_mean = sum((hopping**2)*onsite for hopping, onsite in zip(V_arr, e_arr))
    e_arr = e_arr - (arim_mean)/(2*W)
    max_diff = e_arr[-1] - e_arr[0]
    e_arr = e_arr*(2*W/(max_diff))
    return e_arr        

def hyb(onsite, hopping):
    """
    Returns hybridisation function
    """
    return sum((V_**2)*inverse(iOmega_n - E_) for E_, V_ in zip(onsite, hopping))

def get_data(G, basis):
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
    
def save_me(fname, G, basis, target_n_tau):

    if basis == "iw": 
        data = np.asarray(get_data(G, basis))
        target = 1
        skip_factor = 1
        write_me(fname, skip_factor, data)
                
    if basis == "tau": 
        data = np.asarray(get_data(G, basis))
        skip_factor=int((len(data[0])-1)/(target_n_tau))      
        write_me(fname, skip_factor, data)

    if basis == "legendre":
        data = np.asarray(get_data(G, basis))
        skip_factor=1
        write_me(fname, skip_factor, data)
    
def write_me(filename, skip_factor, data):
    with open(filename, 'a') as f:
        np.savetxt(f,
                   data[1][::skip_factor, np.newaxis].T,
                   delimiter=",",
                   fmt="%1.6f")

def extract_hyb(hyb_param, bath_param):
    """
    Export Delta csv files in both legendre and tau bases
    """

    Delta_iw = GfImFreq(indices=hyb_param["indices"], beta=hyb_param["beta"],
                        n_points=hyb_param["n_iw"], name=r"$G(i \omega_n)$")
    
    E_, V_ = bath_param
    Delta_iw <<  hyb(E_,V_)

    mesh_l = MeshLegendre(beta=hyb_param["beta"], S = "Fermion",
                          n_max=hyb_param["n_l"])
    Delta_leg = GfLegendre(indices=hyb_param["indices"],
                           mesh=mesh_l, name=r'$G_l$')
    Delta_tau = GfImTime(indices=hyb_param["indices"], beta=hyb_param["beta"],
                         n_points=hyb_param["n_tau"], name=r"$\Delta(\tau)$")


    # print(hyb_param["n_tau"])
    # print(hyb_param['target_n_tau'])

    Delta_tau << Fourier(Delta_iw)        
    Delta_leg << MatsubaraToLegendre(Delta_iw)    
    
    data_tau = np.asarray(get_data(Delta_tau, "tau"))

    # skip_factor = int((len(data_tau[0])-1)/(hyb_param["target_n_tau"]))
    # ret_tau = data_tau[1][::skip_factor, np.newaxis].T

    idxs = np.linspace(0, len(data_tau[0]) - 1, hyb_param["target_n_tau"]).astype(int)
    ret_tau=np.zeros((1,len(idxs)))
    for i,idx in enumerate(idxs):
        ret_tau[0][i]=data_tau[1][idx]

    
    data_leg = np.asarray(get_data(Delta_leg, "legendre"))
    ret_leg = data_leg[1][::1, np.newaxis].T
    return ret_tau, ret_leg

def write_me_SERIAL_ed(fname, mpi_data):
    y = np.concatenate(mpi_data)
    arr = y[:,0]
    with open(fname, 'ab') as f:
        np.savetxt(fname, arr, delimiter=",", fmt="%1.6f")
        
def write_me_SERIAL_POM(fname, mpi_data):
    y = np.concatenate(mpi_data)
    conc_ = []
    for i in y[:,:][:]:
        conc_.append(i)
    with open(fname, 'ab') as f:
        np.savetxt(fname, conc_, delimiter=",", fmt="%1.6f")

def write_me_SERIAL_POM_tau(fname, mpi_data):
    y = np.concatenate(mpi_data)
    with open(fname, 'ab') as f:
        np.savetxt(fname, y[:,0], delimiter=",", fmt="%1.6f")

        
