import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
 
def plot_df(ax,df,yname,label):
    ax.plot(df['epoch'].values,df[yname].values,label=label)
    ax.set_xlabel('epoch')
    ax.set_ylabel(yname)
    ax.legend(loc='best')

def plot_df_adv(ax,df,dfkey,yname,label,color,ls='-',marker='',logsclae=False):
    ax.plot(df['epoch'].values,df[dfkey].values,label=label,ls=ls,marker=marker,color=color)
    if logsclae: ax.set_yscale('log')
    ax.set_xlabel('epoch')
    ax.set_ylabel(yname)
    ax.legend(loc='best')

def plot_df_adv2(ax,df,dfkey,yname,yerr,label,color,ls='-',marker='',logsclae=False):
    ax.errorbar(df['epoch'].values,df[dfkey].values,yerr=df[yerr].values,label=label,ls=ls,marker=marker,color=color)

    if logsclae: ax.set_yscale('log')
    ax.set_xlabel('epoch')
    ax.set_ylabel(yname)
    ax.legend(loc='best')

def plot_stepLR(ax,num_epochs=20,lr=0.1,gamma=0.1,step=10,plot_logscale=False):
    num_each_step=100
    num_ponts=num_epochs*num_each_step

    lr_list=np.zeros(num_ponts)
    epochs_list=np.linspace(0,num_epochs,num_ponts)
    for epoch in range(num_epochs):
        lr_list[epoch*num_each_step:(epoch+1)*num_each_step]=lr
        if (epoch+1)%step==0:
            lr=lr*gamma

    ax.plot(epochs_list,lr_list,label=f'StepLR,gamma {gamma}')
    if plot_logscale: ax.set_yscale('log')
    ax.legend()
    ax.set_xlabel('epoch')
    ax.set_ylabel('lr')
    

def plot_num_latents():
    # plot error loss
    num_latents_list=[256,128,64,32,16,8]

    fig, ax = plt.subplots(2,2,figsize=(8,6),dpi=300)

    for num_latent in num_latents_list:
        file_name='model_weights%d.pth'%num_latent
        dictionary_name='dict%d.pkl'%num_latent

        with open(dictionary_name,'rb') as f:
            data_dict=pickle.load(f)

        df=pd.DataFrame(data_dict)

        f_dict = lambda name: [item['%s'%name] for item in data_dict]
        plot_df(ax[0,0],df,'train MSE','num_latent %d'%num_latent)
        plot_df(ax[0,1],df,'test MSE','num_latent %d'%num_latent)
        plot_df(ax[1,0],df,'train MAE','num_latent %d'%num_latent)
        plot_df(ax[1,1],df,'test MAE','num_latent %d'%num_latent)

    fig.suptitle('Uniform Characterisation,adam,lr=0.001,mini-batch=50,dropout=0')
    plt.tight_layout()
    plt.savefig('err_loss.png')

def plot_lr():
    # plot error loss
    learning_rate=[1e-5,1e-6]

    fig, ax = plt.subplots(2,2,figsize=(8,6),dpi=300)

    for lr in learning_rate:
        file_name='model_weights%.5f.pth'%lr
        dictionary_name='dict_AdamW%.5f.pkl'%lr

        with open(dictionary_name,'rb') as f:
            data_dict=pickle.load(f)

        df=pd.DataFrame(data_dict)

        plot_df_adv(ax[0,0],df,'train MSE','train MSE',label='lr %.0e'%lr,logsclae=True)
        plot_df_adv(ax[0,1],df,'test MSE','test MSE',label='lr %.0e'%lr,logsclae=True)
        plot_df_adv(ax[1,0],df,'train MAE','train MAE',label='lr %.0e'%lr,logsclae=True)
        plot_df_adv(ax[1,1],df,'test MAE','test MAE',label='lr %.0e'%lr,logsclae=True)

    fig.suptitle('Uniform Characterisation,wadam,latent=256,mini-batch=50,dropout=0.2')
    plt.tight_layout()
    idx=6
    # while os.path.isfile('err_loss%d.png'%idx):
    #     idx+=1
    plt.savefig('err_loss%d.png'%idx)


def __plot__(ax,df,lr,step,gamma,plot_logscale,label=''):
    plot_df_adv(ax[0,0],df,'train MSE','MSE',label+'train','-','',plot_logscale)
    plot_df_adv(ax[0,0],df,'test MSE','MSE',label+'val','','^',plot_logscale)
    plot_df_adv(ax[1,0],df,'train MAE','MAE',label+'train','-','',plot_logscale)
    plot_df_adv(ax[1,0],df,'test MAE','MAE',label+'val','','^',plot_logscale)
    num_epochs=df['epoch'].max()
    plot_stepLR(ax=ax[0,1],
                num_epochs=num_epochs,
                lr=lr,
                step=step,
                gamma=gamma,
                plot_logscale=plot_logscale)
    return ax

def plot_loss_versus_epoch(file,lr,step,gamma,opt='adamw',num_latent=256,mini_batch=64,plot_logscale=False):

    fig, ax = plt.subplots(2,2,figsize=(8,6),dpi=300)

    with open(file,'rb') as f:
        data_dict=pickle.load(f)

    df=pd.DataFrame(data_dict)

    ax=__plot__(ax,df,lr,step,gamma,plot_logscale)

    title=f'Uniform Characterisation {opt},{num_latent=},{mini_batch=},'
    fig.suptitle(title)
    plt.tight_layout()
    print('saving: err_loss.png')
    plt.savefig('err_loss.png')


