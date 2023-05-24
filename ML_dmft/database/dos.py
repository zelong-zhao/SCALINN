from triqs.gf import *
from triqs.plot.mpl_interface import oplot,plt
from math import pi
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use(plt.style.available[-2])

def G_W(AIM_solved_dict,index=0,solver='ED_KCL'):
    G_l_read=AIM_solved_dict[solver][index]['G_l']
    G_tau_read=AIM_solved_dict[solver][index]['G_tau']

    n_l=AIM_solved_dict[solver][index]['data info']['n_l']
    beta=AIM_solved_dict[solver][index]['data info']['beta']
    n_tau=AIM_solved_dict[solver][index]['data info']['n_tau']
    n_iw=AIM_solved_dict[solver][index]['data info']['n_iw']

    G_l=GfLegendre(indices=[0],beta=beta,n_points = n_l)
    G_l.data[:,:,0] = G_l_read.reshape((len(G_l_read),1))
    G_tau=GfImTime(indices=[0],beta=beta, n_points=10*n_iw+1)

    G_tau.set_from_legendre(G_l)

    G_iw=GfImFreq(indices=[0],
                beta=beta,
                n_points=n_iw)       
                    
    G_iw << Fourier(G_tau)
    G_w= GfReFreq(indices = [0], window = (-8, 8))
    G_w.set_from_pade(G_iw,100, 0.01)

    mesh_=[]
    for t in G_w.mesh:
        mesh_.append(t.value)
    mesh_=np.array(mesh_).reshape((len(mesh_),1))
    
    G_w_out=G_w.data[:,:,0]
    
    Aw=-G_w.imag/pi

    return mesh_,G_w_out

def GW_from_G_tau(AIM_solved_dict,index=0,solver='ED_KCL'):

    G_l_read=AIM_solved_dict[solver][index]['G_l']
    G_tau_read=AIM_solved_dict[solver][index]['G_tau']

    n_l=AIM_solved_dict[solver][index]['data info']['n_l']
    beta=AIM_solved_dict[solver][index]['data info']['beta']
    n_tau=AIM_solved_dict[solver][index]['data info']['n_tau']
    n_iw=AIM_solved_dict[solver][index]['data info']['n_iw']

    tau_mash=np.linspace(0, beta, len(G_tau_read), endpoint=True)

    from scipy.interpolate import UnivariateSpline

    G_tau_fit=UnivariateSpline(tau_mash,G_tau_read)
    G_tau_fit.set_smoothing_factor(1.0e-6)

    tau_mash_fit=np.linspace(0, beta, 1+(n_iw*10), endpoint=True)

    G_tau=GfImTime(indices=[0],beta=beta, n_points=1+(n_iw*10))

    G_tau.data[:,:,0]=G_tau_fit(tau_mash_fit).reshape((len(tau_mash_fit),1))

    G_iw=GfImFreq(indices=[0],
                beta=beta,
                n_points=n_iw) 
                          
    G_iw << Fourier(G_tau)
    G_w= GfReFreq(indices = [0], window = (-8, 8))
    G_w.set_from_pade(G_iw)

    mesh_=[]
    for t in G_w.mesh:
        mesh_.append(t.value)
    mesh_=np.array(mesh_).reshape((len(mesh_),1))
    
    G_w_out=G_w.data[:,:,0]
    
    Aw=-G_w.imag/pi

    return mesh_,G_w_out

def density_of_states(AIM_solved_dict,index=0,solver='ED_KCL'):
    mesh,G_Wout=G_W(AIM_solved_dict,index=index,solver=solver)
    # mesh,G_Wout=GW_from_G_tau(AIM_solved_dict,index=index,solver=solver)

    Aw=-G_Wout.imag/pi
    plot_dos(mesh,Aw)
    return mesh,Aw

def plot_dos(mesh,Aw):
    fig, ax = plt.subplots()
    ax.plot(mesh,Aw)
    ax.set_xlim(-4,4)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('$\omega$(eV)')
    ax.set_ylabel('A($\omega$)')
    fig.savefig('Aw.png',dpi=100)