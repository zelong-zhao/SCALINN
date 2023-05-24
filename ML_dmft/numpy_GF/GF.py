# -*- coding: utf-8 -*-
"""
Green Functions
===============
Interface to Green's function of imaginary time and imaginary frequency 

# Modifed From https://github.com/Titan-C/pydmft/tree/master/dmft
"""

import numba as nb
import numpy as np
from numpy.fft import fft, ifft

def real_to_complex(z):      # real vector of length 2n -> complex of length n
    return z[:len(z)//2] + 1j * z[len(z)//2:]

def complex_to_real(z):      # complex vector of length n -> real of length 2n
    return np.concatenate((np.real(z), np.imag(z)))

# @nb.jit(nopython=True,cache=True)
def matsubara_freq(beta:np.float64,n_w:np.int8,Positive_Semi_defined=True)->np.ndarray:
    r"""
    Ferminoic Matsubara Frequency return Omega_n
    """
    Omega_n_semi_defined=(np.pi * (1 + 2 * np.arange(n_w)) / beta).reshape(n_w,1)

    if Positive_Semi_defined:
        Omega_n=Omega_n_semi_defined

    else:
        Omega_n=np.zeros((2*n_w,1),dtype=np.float64)
        Omega_n[n_w:2*n_w,0]=Omega_n_semi_defined[:,0]
        Omega_n[0:n_w,0]=np.flip(Omega_n_semi_defined[:,0])*(-1)

    return Omega_n


# @nb.jit(nopython=True,cache=True)
def greenF(Omega_n:np.ndarray, sigma:np.int8=0, mu:np.float64=0, D:np.int8=1)->np.ndarray:
    r"""Calculate the Bethe lattice Green function, defined as part of the
    hilbert transform.

    :math:`G(i\omega_n) = \frac{2}{i\omega_n + \mu - \Sigma + \sqrt{(i\omega_n + \mu - \Sigma)^2 - D^2}}`

    Parameters
    ----------
    w_n : real float array
            fermionic matsubara frequencies.
    sigma : complex float or array
            local self-energy
    mu : real float
            chemical potential
    D : real
        Half-bandwidth of the bethe lattice non-interacting density of states

    Returns
    -------
    complex ndarray
            Interacting Greens function in matsubara frequencies, all odd
            entries are zeros
    """
    zeta = 1.j * Omega_n + mu - sigma
    sq = np.sqrt((zeta)**2 - D**2)
    sig = np.sign(sq.imag * Omega_n)
    out=2. / (zeta + sig * sq)
    return out.reshape(len(Omega_n),1)

# @nb.jit(nopython=True,cache=True)
def tau_wn_setup(beta:float,n_w:int,n_tau:np.int8=10241,Positive_Semi_defined:bool=True):
    r"""
    tau wn setup tau in shape of 10*n_w +1
    """
    Omega_n = matsubara_freq(beta=beta,n_w=n_w,Positive_Semi_defined=Positive_Semi_defined)
    tau = np.linspace(0,beta,n_tau).reshape(n_tau,1) #(0,beta]
    return tau,Omega_n

# @nb.jit(nopython=True,cache=True)
def tail(Omega_n:np.ndarray, coef:np.ndarray, powers:np.ndarray)->np.ndarray:
    r"""
    ----
    Input:
    Omega_n:np.ndarray float

    -----
    Output:
    tail:np.ndarray float
    """
    out_tail=np.zeros((len(Omega_n),1))
    for idx,w_n in enumerate(Omega_n):
        for c, p in zip(coef, powers):
            out_tail[idx]+=c/w_n**p
    return out_tail

# @nb.jit(nopython=True,cache=True)
def tail_coef(Omega_n, data, powers):
    Omega_n=Omega_n[:,0]
    A = np.zeros((len(Omega_n),len(powers)))
    for index,p in enumerate(powers):
        for idx,w_n in enumerate(Omega_n):
            A[idx,index]=np.array(1/w_n**p)
    fitted_coef=np.linalg.lstsq(A, data,rcond=-1)[0]
    fitted_coef=fitted_coef.flatten()
    return fitted_coef

def tail_coef_tau(tau,data,powers,x:int,span:int):
    powers=[1,2]
    data=data[-x:-x + span]
    tau=tau.flatten()[-x:-x + span]
    A = np.zeros((len(tau),len(powers)))
    def time_tail(tau,power):
        beta = tau[1] + tau[-1]
        if power == 1:
            return -1/2
        if power == 2:
            return (1/2)*(tau-beta/2)

    for index,p in enumerate(powers):
        A[:,index]=time_tail(tau,p)
    fitted_coef=np.linalg.lstsq(A, data,rcond=-1)[0]
    return fitted_coef.flatten()


# @nb.jit(nopython=True,cache=True)
def lstsq_tail_fit(Omega_n, inp_gf, x, span=30):
    """Perform a Least squares fit to the tail of Green function

    the fit is done in inp_gf[-x:-x + span]

    Parameters:
        w_n (real 1D ndarray) : Matsubara frequencies
        inp_gf (complex 1D ndarray) : Green function to fit
        x (int) : counting from last element from where to do the fit
        span (int) : amount of frequencies to do the fit over
        negative_freq (bool) : Array has negative Matsubara frequencies
    Returns:
        complex 1D ndarray : Tail patched Green function (copy of origal)
        real 1D ndarray : The First 2 moments
"""
    tw_n = Omega_n[-x:-x + span].copy()
    datar = inp_gf[-x:-x + span].real.copy()
    datai = inp_gf[-x:-x + span].imag.copy()

    re_c = tail_coef(tw_n, datar, np.array([2]))
    re_tail = tail(Omega_n, re_c, np.array([2]))
    im_c = tail_coef(tw_n, datai, np.array([1,3]))
    im_tail = tail(Omega_n, im_c, np.array([1,3]))

    f_tail = re_tail + 1j * im_tail

    patgf = inp_gf.copy()

    patgf[-x:,0] = f_tail[-x:,0]

    return patgf, np.array([im_c[0], re_c[0],im_c[1]]).flatten()


# @nb.jit(nopython=True,cache=True)
def freq_tail_fourier_orderIII(tail_coef, beta, tau, Omega_n):
    r"""Fourier transforms analytically the slow decaying tail_coefs of
    the Greens functions [matsubara]_

    +------------------------+-----------------------------------------+
    | :math:`G(iw)`          | :math:`G(t)`                            |
    +========================+=========================================+
    | :math:`(i\omega)^{-1}` | :math:`-\frac{1}{2}`                    |
    +------------------------+-----------------------------------------+
    | :math:`(i\omega)^{-2}` | :math:`\frac{1}{2}(\tau-\beta/2)`       |
    +------------------------+-----------------------------------------+
    """

    freq_tail = tail_coef[0] / (1.j * Omega_n)\
        + tail_coef[1] / (1.j * Omega_n)**2\
        + tail_coef[2] / (1.j * Omega_n)**3

    time_tail = - tail_coef[0] / 2  \
        + tail_coef[1] / 2 * (tau - beta / 2) \
                - tail_coef[2] / 4 * (tau**2 - beta * tau)

    return freq_tail.reshape(len(freq_tail),1), time_tail.reshape(len(time_tail),1)

def freq_tail(tail_coef, Omega_n):
    r"""Fourier transforms analytically the slow decaying tail_coefs of
    the Greens functions [matsubara]_

    +------------------------+-----------------------------------------+
    | :math:`G(iw)`          | :math:`G(t)`                            |
    +========================+=========================================+
    | :math:`(i\omega)^{-1}` | :math:`-\frac{1}{2}`                    |
    +------------------------+-----------------------------------------+
    | :math:`(i\omega)^{-2}` | :math:`\frac{1}{2}(\tau-\beta/2)`       |
    +------------------------+-----------------------------------------+
    """

    freq_tail = tail_coef[0] / (1.j * Omega_n)\
        + tail_coef[1] / (1.j * Omega_n)**2

    return freq_tail.reshape(len(freq_tail),1)

def freq_tail_fourier(tail_coef, beta, tau, Omega_n):
    r"""Fourier transforms analytically the slow decaying tail_coefs of
    the Greens functions [matsubara]_

    +------------------------+-----------------------------------------+
    | :math:`G(iw)`          | :math:`G(t)`                            |
    +========================+=========================================+
    | :math:`(i\omega)^{-1}` | :math:`-\frac{1}{2}`                    |
    +------------------------+-----------------------------------------+
    | :math:`(i\omega)^{-2}` | :math:`\frac{1}{2}(\tau-\beta/2)`       |
    +------------------------+-----------------------------------------+
    """

    freq_tail = tail_coef[0] / (1.j * Omega_n)\
        + tail_coef[1] / (1.j * Omega_n)**2\
        # + tail_coef[2] / (1.j * Omega_n)**3

    time_tail = - tail_coef[0] / 2  \
        + tail_coef[1] / 2 * (tau - beta / 2) \
                # - tail_coef[2] / 4 * (tau**2 - beta * tau)


    return freq_tail.reshape(len(freq_tail),1), time_tail.reshape(len(time_tail),1)
    
def gw_invfouriertrans(g_iwn, tau, Omega_n, tail_coef=(1., 0., 0.)):
    tau=tau.flatten()
    beta = tau[1] + tau[-1]

    freq_tail, time_tail = freq_tail_fourier(tail_coef, beta, tau, Omega_n)
    giwn = g_iwn - freq_tail
    
    g_tau = fft(giwn[:,0], len(tau)) * np.exp(-1j * np.pi * tau / beta)
    g_tau = g_tau.reshape(len(g_tau),1)

    return (g_tau * 2 / beta).real + time_tail

def gt_fouriertrans(g_tau, tau, Omega_n, tail_coef=(1., 0., 0.)):
    tau=tau.flatten()
    beta = tau[1] + tau[-1]
    freq_tail, time_tail = freq_tail_fourier(tail_coef, beta, tau, Omega_n)

    gtau = g_tau - time_tail

    gtau=gtau.flatten()

    freq_tail=freq_tail.flatten()

    g_iw=beta * ifft(gtau * np.exp(1j * np.pi * tau / beta))[..., :len(Omega_n)] + freq_tail
    g_iw=g_iw.reshape(len(g_iw),1)
    return g_iw

def hyb_np(onsite,hopping,n_iw,beta):
    """
    only diagonal part
    """
    iOmega_n=1j*matsubara_freq(beta=beta,n_w=n_iw)
    hyb=np.zeros((len(iOmega_n),1),dtype=complex)
    for index,iwn in enumerate(iOmega_n):
        for E_, V_ in zip(onsite, hopping):
            hyb[index]+=(V_**2)/(iwn - E_)
    return hyb

def iwn_distance_func(delta1,delta2,Omega_n,omega_c=100):
    """
    $d=\sum_n ｜delta(iwn) - delta_2(iwn)｜^2 /w_n$
    """
    sumd=0
    for i in range(0,omega_c):
        diff=np.abs(delta1[i]-delta2[i])**2
        diff=diff/Omega_n[i]
        sumd=diff+sumd
    return np.float(sumd)

def iw_obj_flip(iw_obj_in):
    """
    iw_obj_flip
    re(iw_obj)[1024:2048]=re(iw_obj_in)[0:]
    """
    dim=len(iw_obj_in)
    dtype=iw_obj_in.dtype
    iwb_obj_out=np.zeros((int(2*dim),1),dtype=dtype)
    iwb_obj_out[dim:2*dim]=iw_obj_in
    iwb_obj_out[0:dim].real=np.flip(iw_obj_in.real)
    iwb_obj_out[0:dim].imag=np.flip(iw_obj_in.imag)*(-1)
    return iwb_obj_out

def Gtau_expand(beta,target_num_tau,G_tau):
    import scipy.interpolate
    num_tau=len(G_tau)
    tau_data=np.linspace(0,beta,num_tau,dtype=float)
    G_tau_fit=scipy.interpolate.UnivariateSpline(tau_data,G_tau,s=0)
    tau_out=np.linspace(0,beta,target_num_tau,dtype=float)
    return tau_out.reshape(target_num_tau,1),G_tau_fit(tau_out).reshape(target_num_tau,1)

def format_bath_for_ced_ed(e_list, V_list):

    bath=np.concatenate((V_list,e_list))
    # print("------ BATH FORMAT FOR CED SOLVER IS -------")
    # print(bath)
    return bath

def quasi_partical_weight(sigma_iw,beta):
    Z = 1 / (1 - (sigma_iw[0].imag * beta / np.pi))
    return Z

def len_num_imp_ced_ed(bath:np.ndarray,upper_lim_search:int=1000000):
    """
    input bath
    V1 V2 V3 V4 e11 e12 e13 e14 e22 e23 e24 e33 e34 e44

    output num imp
    """
    for num_imp in range(1,upper_lim_search):
        bath_test=format_bath_for_ced_ed(np.zeros(num_imp), np.zeros(num_imp))
        if len(bath) == len(bath_test):
            return num_imp

def inverse_format_bath_for_ced_ed_diagnoal_only(bath):
    """
    format of ed.fit.param
    V1 V2 e11 e12 e22
    V1 V2 V3 e11 e12 e13 e22 e23 e33
    V1 V2 V3 V4 e11 e12 e13 e14 e22 e23 e24 e33 e34 e44
    """
    bath=np.array(bath).flatten()
    num_imp=len_num_imp_ced_ed(bath)
    V_list=bath[0:num_imp]
    e_list=bath[num_imp:]

    e_mat=np.zeros((num_imp,num_imp))

    # for i in range(len(e_list)):
    # 	e_mat[i][i] = e_list[i]

    counter=0
    for i in range(num_imp):
        for j in range(num_imp):
            if i==0: 
                e_mat[i,j]=e_list[counter]
                counter+=1
            else:
                if j>=i:
                    e_mat[i,j]=e_list[counter]
                    counter+=1

    e_list=np.array([e_mat[i,i] for i in range(num_imp)]).flatten()
    return e_list, V_list



def inverse_format_bath_for_ced_ed_eps_matrix(bath):
    """
    format of ed.fit.param
    V1 V2 e11 e12 e22
    V1 V2 V3 e11 e12 e13 e22 e23 e33
    V1 V2 V3 V4 e11 e12 e13 e14 e22 e23 e24 e33 e34 e44

    output 
    """
    bath=np.array(bath).flatten()
    num_imp=len_num_imp_ced_ed(bath)
    V_list=np.array(bath[0:num_imp]).flatten()
    e_list=bath[num_imp:]

    e_mat=np.zeros((num_imp,num_imp))

    # for i in range(len(e_list)):
    # 	e_mat[i][i] = e_list[i]

    counter=0
    for i in range(num_imp):
        for j in range(num_imp):
            if i==0: 
                e_mat[i,j]=e_list[counter]
                e_mat[j,i]=e_list[counter]
                counter+=1
            else:
                if j>=i:
                    e_mat[i,j]=e_list[counter]
                    e_mat[j,i]=e_list[counter]
                    counter+=1

    return e_mat,V_list