from tkinter import font
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_one_pkl(y_axis,basis,labels, beta, target_n_tau, n_l,file_name='test'):
    matplotlib.style.use(plt.style.available[-2])
    plt.figure()
    fig, ax = plt.subplots(1, 1,figsize=(8,6))

    if basis == "tau" or basis =='G_tau': 
        x_axis = np.linspace(0, beta, target_n_tau, endpoint=True)
        dots= None

    elif basis == "legendre" or basis == 'G_l': 
        x_axis = np.linspace(1, n_l, n_l, endpoint = True)    
        key='G_l'
        dots= "o"
    
    elif basis == "iw" or basis == 'G_iw': 
        length=y_axis[0].shape[1]
        x_axis = np.arange(-length/2.,length/2.,dtype=int)
        key='G_iw'
        dots= "."
    
    else:
        raise Exception('Basis wrong')

    ax=plot_from_pkl(ax,y_axis, x_axis, labels,dots,basis)

    if basis == "tau" or basis =='G_tau':
        ax.set_xlabel(r"$\tau$",fontsize=15)
        ax.set_ylabel(r"$G(\tau)$",fontsize=15)
        fig.tight_layout()
        fig.savefig(file_name+'.png',dpi=300)

    elif basis == "legendre" or basis == 'G_l': 
        ax.set_xlabel(r"l",fontsize=15)
        ax.set_ylabel(r"$G_l$",fontsize=15)
        fig.tight_layout()
        fig.savefig(file_name+'.png',dpi=300)

    elif basis == "iw" or basis == 'G_iw': 
        ax.set_xlabel(r"iwn",fontsize=15)
        ax.set_ylabel(r"$G_(iwn)$",fontsize=15)
        fig.tight_layout()
        fig.savefig(file_name+'.png',dpi=300)
    

    return ax

def plot_from_pkl(ax,y_axis, x_axis, labels,dots,basis):
    for y,label in zip(y_axis,labels):
        # shape=y.shape[1]
        # y=y.reshape((shape,))
        if basis=='G_iw':
            y=y.imag.flatten()
        else:
            y=y.flatten()
        ax.plot(x_axis[::1],
                y[::1],
                label = label,
                marker=dots)
    ax.legend(fontsize=15)
    return ax