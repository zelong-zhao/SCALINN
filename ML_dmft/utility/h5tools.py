from re import T
from h5 import HDFArchive
import numpy as np
from ML_dmft.utility.db_gen import *
from ML_dmft.utility.tools import *
import matplotlib
import matplotlib.pyplot as plt
import os,pickle

def h52diction(prefix,AIM):
    fname=os.path.join(prefix,'db',AIM+'_backup.h5')
    print('loading:',fname)
    data_point_=[]
    with HDFArchive(fname,'r') as ar:
        for idx in ar:
            data_strct={}
            for item in ar[idx]:
                data_strct[item]=ar[idx][item]
            data_point_.append(data_strct)
    return data_point_

def h52diction_values(H5_data,AIM,target_n_tau):
    if AIM == 'IPT':
        data_point_=[]
        for item in H5_data:
            out_dic=IPT_H5_data(item,target_n_tau=target_n_tau)
            data_point_.append(out_dic)
        return data_point_
    if AIM == 'ED_KCL':
        data_point_=[]
        for item in H5_data:
            out_dic=ED_KCL_data(item,target_n_tau=target_n_tau)
            data_point_.append(out_dic)
        return data_point_

def H52save2pkl(AIM,target_n_tau,prefix='./'):
    data_point_=h52diction(prefix=prefix,AIM=AIM)
    data_point=h52diction_values(H5_data=data_point_,AIM=AIM,target_n_tau=target_n_tau)
    path_pkl=os.path.join(prefix,'db',AIM+'_solved.pkl')
    print(path_pkl,'is created')
    with open(path_pkl, 'wb') as f:
        pickle.dump(data_point, f)
    

def ED_KCL_data(input_dic,target_n_tau):
    out_dic={}
    out_dic['AIM params']=input_dic['AIM params']

    G_l=input_dic['G_l']
    data_leg = np.asarray(get_data(G_l, "legendre"))
    ret_leg = data_leg[1]
    ret_leg = ret_leg.reshape((1,1,len(ret_leg)))
    ret_leg=np.concatenate(ret_leg)
    out_dic['G_l']=ret_leg

    G_tau=input_dic['G_tau']
    data_tau = np.asarray(get_data(G_tau, "tau"))
    skip_factor=int((len(data_tau[0])-1)/(target_n_tau))
    ret_tau = data_tau[1][::skip_factor, np.newaxis].T,
    out_dic['G_tau'] = np.concatenate(ret_tau)

    out_dic['logfile-p1_DC']=input_dic['logfile-p1_DC']

    return out_dic


# def extract_data(self):

#     data_tau = np.asarray(get_data_basis(self.G_tr, "tau"))
#     skip_factor=int((len(data_tau[0])-1)/(self.target_n_tau))
#     ret_tau = data_tau[1][::skip_factor, np.newaxis].T,
#     data_leg = np.asarray(get_data_basis(self.G_l, "legendre"))
#     ret_leg = data_leg[1]
#     ret_leg = ret_leg.reshape((1,1,len(ret_leg)))
#     return ret_tau, ret_leg


def IPT_H5_data(input_dic,target_n_tau):
    # print(input_dic)
    out_dic={}
    out_dic['AIM params']=input_dic['AIM params']

    G_l=input_dic['G_l']
    data_leg = np.asarray(get_data(G_l, "legendre"))
    ret_leg = data_leg[1][::1, np.newaxis].T,
    out_dic['G_l']=np.concatenate(ret_leg)

    G_tau=input_dic['G_tau']
    data_tau = np.asarray(get_data(G_tau, "tau"))
    skip_factor=int((len(data_tau[0])-1)/(target_n_tau))
    ret_tau = data_tau[1][::skip_factor, np.newaxis].T,
    out_dic['G_tau'] = np.concatenate(ret_tau)
    return out_dic

# def extract_data(self):
#     data_tau = np.asarray(get_data(self.G_tau, "tau"))
#     skip_factor=int((len(data_tau[0])-1)/(self.target_n_tau))
#     ret_tau = data_tau[1][::skip_factor, np.newaxis].T,
#     data_leg = np.asarray(get_data(self.G_l, "legendre"))
#     ret_leg = data_leg[1][::1, np.newaxis].T,
#     return ret_tau, ret_leg

def comp_H5(data_dict,plot_param, index, basis, beta, target_n_tau, n_l):
    """
    For PT solver plot a sample of the results
    """
    if basis == "tau": 
        x_axis = np.linspace(0, beta, target_n_tau+1, endpoint=True)
        key='G_tau'     
    if basis == "legendre":
        x_axis = np.linspace(1, n_l, n_l, endpoint = True)    
        key='G_l'

    whoami = mpi_rank()
    if whoami == plot_param["chosen_rank"]:
        #fig,axes = plt.subplots(1)
        cm2inc=1./2.54
        fig,axes = plt.subplots(1,
                                figsize=(11.0*cm2inc,8.0*cm2inc),
                                dpi=100)
        colors = plt.cm.tab20b(np.linspace(0,1,4))
        
        count = 0 
        for i in data_dict:
            y_axis = data_dict[i][index][key]

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
            axes = plot_from_h5(y_axis, x_axis, index, i_new, axes, plot_param, colors[count],
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
   
def plot_from_h5(y_axis, x_axis, index, descrip, axes, hyb_param, cin, ms, mt, skip, dots):

    shape=y_axis.shape[1]
    y_axis=y_axis.reshape((shape,))

    if hyb_param["basis"] == "legendre" and hyb_param["poly_semilog"]:        
        axes.semilogy(x_axis, np.abs(y_axis),
                      '-o', label = descrip)
        axes.set_ylim([1e-6,1e+1])
    else:
        axes.plot(x_axis[::skip], y_axis[::skip],
                  markersize=ms,
                  linestyle=mt,
                  marker=dots,
                  color=cin,
                  label = descrip)        
    return axes