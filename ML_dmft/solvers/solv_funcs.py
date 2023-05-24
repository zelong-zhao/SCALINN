import numpy as np
import pandas as pd
from ML_dmft.utility.db_gen import *
from ML_dmft.utility.tools import * 

def gf_to_triqs(g,b_v,nomg,label):
    #mesh_iw = MeshImFreq(beta, 'Fermion', n_max=1000)
    gf=GfImFreq(indices=[0], beta=b_v, n_points = nomg, name=label)
    # print(len(gf.data))
    # ES BUG: this only populates the imaginary part now!!
    gf.data[:,0,0] = 0.0+ 1j*g 
    # get_obj_prop(gf)
    return gf
           
def do_pade(gf,L,eta,wmin,wmax):
    
    g_pade = GfReFreq(indices = [0], window = (wmin,wmax))                       
    g_pade.set_from_pade(gf, n_points = L, freq_offset = eta)

    return g_pade

    
def comp_zz(AIM, plot_param, index, basis, beta, target_n_tau, n_l,prefix):
    """
    For PT solver plot a sample of the results
    """
    if basis == "tau": 
        x_axis = np.linspace(0, beta, target_n_tau+1, endpoint=True)        
    if basis == "legendre":
        x_axis = np.linspace(1, n_l, n_l, endpoint = True)        
    
    whoami = mpi_rank()
    if whoami == plot_param["chosen_rank"]:
        #fig,axes = plt.subplots(1)
        cm2inc=1./2.54
        fig,axes = plt.subplots(1,
                                figsize=(11.0*cm2inc,8.0*cm2inc),
                                dpi=250)
        colors = plt.cm.tab20b(np.linspace(0,1,4))
        
        count = 0 
        for i in AIM:
            y_axis = label_zz("G_"+i, basis,prefix=prefix)

            if i == "IPT":
                i_new = r"IPT"
                ms = 5
                mt = '-'
                skip = 1

            if i=="ED_KCL":
                i_new='ED KCL'
                ms=5
                mt='-.'
                skip = 1

            if basis== "legendre":
                dots= "o"
            else:
                dots= None
            axes = plot_from_csv(y_axis, x_axis, index, i_new, axes, plot_param, colors[count],
                                 ms, mt, skip, dots)
            count = count + 1
        axes.legend()
        #axes.set_title("Comparison between solutions on rank = "+ str(rank))
        if basis == "tau": 
            axes.set_xlabel(r"$\tau$")
            axes.set_ylabel(r"$G(\tau)$")
            fig.tight_layout()
            fig.savefig("gtau.pdf",format='pdf')
        if basis == "legendre":
            axes.set_xlabel(r"l")
            axes.set_ylabel(r"$G_l$")
            fig.tight_layout()
            fig.savefig("gl.pdf",format='pdf')
    return axes



def read_AIM_database_csv(AIM, index, basis, beta, target_n_tau, n_l, prefix):
    """
    For PT solver plot a sample of the results
    """
    y_axis = label_zz("G_"+AIM, basis,prefix=prefix)
    out=extract_from_csv(y_axis,index)
    return out

def delete_files(solver, beta):
    tau_filename = label(solver, "tau")
    leg_filename = label(solver, "legendre")
    del_file(tau_filename)
    del_file(leg_filename)