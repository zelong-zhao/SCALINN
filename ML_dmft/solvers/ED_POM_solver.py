import numpy as np
import time
from triqs.gf import *
from triqs.dos import *
from triqs.operators import Operator, c, c_dag, n
from triqs.utility import mpi
from triqs.utility.comparison_tests import *
from pomerol2triqs import PomerolED
from triqs.plot.mpl_interface import oplot, plt
from d3mft.utility.db_gen import *
from d3mft.utility.tools import *
from itertools import product

class ED_POM_solver():
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
        self.basis = solver_params["basis"]
        self.file_to_write = solver_params["write"]
        self.target_n_tau = solver_params["target_n_tau"]

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
        self.G_tr = self.G_tau_prev.copy()        
        self.imtime_target = np.linspace(0, self.beta, 
                                         self.target_n_tau+1, endpoint=True)

        # Legendre
        self.G_l = GfLegendre(indices = self.indices, beta = self.beta,
                              n_points = self.n_l)        
        self.legendre_target = np.linspace(1, self.n_l, self.n_l, endpoint = True)


    def solve_dirty(self, param):
        """
         Solve the AIM using ED pomerol
        """
        e_list = param["E_p"]
        V_list = param["V_p"]
        self.eps = param["eps"]
        self.U = param["U"]
        mu = self.eps
        U = self.U
        beta = self.beta
        n_iw = self.n_iw
        n_tau = self.n_tau
        spin_names = ("up", "dn")
        # GF structure
        gf_struct = [['up', [0]], ['dn', [0]]]
        # Conversion from TRIQS to Pomerol notation for operator indices
        index_converter = {}
        index_converter.update({(sn, 0) : ("loc", 0, "down" if sn == "dn" else "up") for sn in spin_names})
        index_converter.update({("B%i_%s" % (k, sn), 0) : ("bath" + str(k), 0, "down" if sn == "dn" else "up")
                                for k, sn in product(list(range(len(e_list))), spin_names)})

        # Make PomerolED solver object
        ed = PomerolED(index_converter, verbose = True)

        # Number of particles on the impurity
        H_loc = mu*(n('up', 0) + n('dn', 0)) + U * (n('up', 0) -0.5) * (n('dn', 0) -0.5) 

        # Bath Hamiltonian
        H_bath = sum(eps*n("B%i_%s" % (k, sn), 0)
                     for sn, (k, eps) in product(spin_names, enumerate(e_list)))

        # Hybridization Hamiltonian
        H_hyb = Operator()
        for k, v in enumerate(V_list):
            H_hyb += sum(        v   * c_dag("B%i_%s" % (k, sn), 0) * c(sn, 0) +
                                 np.conj(v)  * c_dag(sn, 0) * c("B%i_%s" % (k, sn), 0)
                                 for sn in spin_names)
        
        # Complete Hamiltonian
        H = H_loc + H_hyb + H_bath

        # Diagonalize H
        ed.diagonalize(H)

        # Compute G(i\omega)
        G_iw = ed.G_iw(gf_struct, beta, n_iw)

        # Compute G(\tau)
        self.G_tau = ed.G_tau(gf_struct, beta, n_tau)
        self.G_tau = self.G_tau['up']
        self.G_l << MatsubaraToLegendre(G_iw['up'])
    
    def extract_data(self):
        
        data_tau = np.asarray(get_data_basis(self.G_tau, "tau"))
        skip_factor=int((len(data_tau[0])-1)/(self.target_n_tau))
        ret_tau = data_tau[1][::skip_factor, np.newaxis].T,
        data_leg = np.asarray(get_data_basis(self.G_l, "legendre"))
        ret_leg = data_leg[1][::1, np.newaxis].T
        return ret_tau, ret_leg

    def construct_hamiltonian(self, spin_names):
        # Number of particles on the impurity
        H_loc = self.eps*(n('up', 0) + n('dn', 0)) + self.U * (n('up', 0) -0.5) * (n('dn', 0) -0.5) 
            
        # Bath Hamiltonian
        H_bath = sum(self.eps*n("B%i_%s" % (k, sn), 0)
                              for sn, (k, eps) in product(spin_names, enumerate(self.e_list)))
            
        # Hybridization Hamiltonian
        H_hyb = Operator()
        for k, v in enumerate(self.V_list):
            H_hyb += sum(v*c_dag("B%i_%s" % (k, sn), 0) * c(sn, 0) +
                         np.conj(v)  * c_dag(sn, 0) * c("B%i_%s" % (k, sn), 0)
                         for sn in spin_names)

        self.H = H_loc + H_hyb + H_bath        
        return self.H

            
def pom_to_triqs(e_list, spin_names):
    # GF structure
    gf_struct = [['up', [0]], ['dn', [0]]]
    # Conversion from TRIQS to Pomerol notation for operator indices
    index_converter = {}
    index_converter.update({(sn, 0) : ("loc", 0, "down" if sn == "dn" else "up") for sn in spin_names})
    index_converter.update({("B%i_%s" % (k, sn), 0) : ("bath" + str(k), 0, "down" if sn == "dn" else "up")
                            for k, sn in product(list(range(len(e_list))), spin_names)})
    return gf_struct, index_converter

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
