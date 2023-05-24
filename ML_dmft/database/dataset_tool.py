import pandas as pd
from ML_dmft.database.dataset import AimG_Dataset_RAM,AIM_dataset_meshG,AIM_Dataset
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from ML_dmft.utility.tools import dump_to_yaml
import scipy.stats
matplotlib.style.use(plt.style.available[-2])
import numpy as np

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def meansure_stats(root,db,solver,basis):
    dataset=AimG_Dataset_RAM(root,db,solver,basis,imag_only=True)
    giw = np.array([dataset.G[i] for i in range(len(dataset))])

    # measure G
    # mean_giw = np.mean(giw,axis=0)
    # std_giw = np.std(giw,axis=0)
    giw_mean=np.mean(giw)
    giw_std=np.std(giw)
    giw_max = np.max(giw)
    giw_min = np.min(giw)

    #standardised
    standardised_giw = (giw-giw_mean)/giw_std
    standardised_giw_max = np.max(standardised_giw)
    standardised_giw_min = np.min(standardised_giw)

    #normalised
    normalised_giw = (giw-giw_min)/(giw_max-giw_min)
    normalised_giw_mean = np.mean(normalised_giw)
    normalised_giw_std = np.std(normalised_giw)
    normalised_giw_max = np.max(normalised_giw)
    normalised_giw_min = np.min(normalised_giw)

    fname=f'./db/{solver}_{basis}_stas.ymal'
    out_dict={f'{basis}_mean':float(giw_mean),
            f'{basis}_std':float(giw_std),
            f'{basis}_max':float(giw_max),
            f'{basis}_min':float(giw_min),
            'standardised_giw_max':float(standardised_giw_max),
            'standardised_giw_min':float(standardised_giw_min),
            'normalised_giw_mean':float(normalised_giw_mean),
            'normalised_giw_std':float(normalised_giw_std),
            'normalised_giw_max':float(normalised_giw_max),
            'normalised_giw_min':float(normalised_giw_min)
            }
    for item in out_dict:print(f"{item}: {out_dict[item]}")
    dump_to_yaml(fname,out_dict)


def meansure_aim_param_stats(root,db,solver):
    dataset=AIM_Dataset(root,db,solver,load_aim_csv_cpu=True)
    data_distribution_list=[]
    E_k_list,V_k_list = [],[]

    for item in dataset.aim_params_csv:
        data_distribution={}        
        data_distribution['U']=item['U']
        data_distribution['eps']=item['eps']
        E_k_list.extend(item['E_p'])
        V_k_list.extend(item['V_p'])
        data_distribution_list.append(data_distribution)

    aim_pd_data=pd.DataFrame(data_distribution_list)

    del data_distribution_list

    out_dict={}
    for item in aim_pd_data:
        out_dict[f"{item}_mean"]=aim_pd_data[item].mean()
        out_dict[f"{item}_max"]=aim_pd_data[item].max()
        out_dict[f"{item}_min"]=aim_pd_data[item].min()
        out_dict[f"{item}_std"]=aim_pd_data[item].std()
        out_dict[f"standardised_{item}_max"]=pd_standardlise(aim_pd_data[item]).max()
        out_dict[f"standardised_{item}_min"]=pd_standardlise(aim_pd_data[item]).min()

    merged_imp = np.concatenate((aim_pd_data['U'].values,aim_pd_data['eps'].values))
    del aim_pd_data

    merged_imp = pd.DataFrame(dict(U_eps_tot=merged_imp))
    for item in merged_imp:
        out_dict[f"{item}_mean"]=merged_imp[item].mean()
        out_dict[f"{item}_max"]=merged_imp[item].max()
        out_dict[f"{item}_min"]=merged_imp[item].min()
        out_dict[f"{item}_std"]=merged_imp[item].std()
        out_dict[f"standardised_{item}_max"]=pd_standardlise(merged_imp[item]).max()
        out_dict[f"standardised_{item}_min"]=pd_standardlise(merged_imp[item]).min()
    del merged_imp


    E_k_list = np.array(E_k_list)
    V_k_list = np.array(V_k_list)

    merged_hyb = np.concatenate((E_k_list,V_k_list))
    merged_hyb = pd.DataFrame(dict(E_k_V_k_tot=merged_hyb))
    for item in merged_hyb:
        out_dict[f"{item}_mean"]=merged_hyb[item].mean()
        out_dict[f"{item}_max"]=merged_hyb[item].max()
        out_dict[f"{item}_min"]=merged_hyb[item].min()
        out_dict[f"{item}_std"]=merged_hyb[item].std()
        out_dict[f"standardised_{item}_max"]=pd_standardlise(merged_hyb[item]).max()
        out_dict[f"standardised_{item}_min"]=pd_standardlise(merged_hyb[item]).min()
    del merged_hyb


    hyb_pd_data = pd.DataFrame(dict(E_k=E_k_list,V_k=V_k_list))
    del E_k_list; del V_k_list; 

    for item in hyb_pd_data:
        out_dict[f"{item}_mean"]=hyb_pd_data[item].mean()
        out_dict[f"{item}_max"]=hyb_pd_data[item].max()
        out_dict[f"{item}_min"]=hyb_pd_data[item].min()
        out_dict[f"{item}_std"]=hyb_pd_data[item].std()
        out_dict[f"standardised_{item}_max"]=pd_standardlise(hyb_pd_data[item]).max()
        out_dict[f"standardised_{item}_min"]=pd_standardlise(hyb_pd_data[item]).min()


    del hyb_pd_data
    for item in out_dict:out_dict[item]=np.float(out_dict[item])
    for item in out_dict:print(f"{item}: {out_dict[item]}")
    
    fname=f'./db/{solver}_aim_params_stas.ymal'
    dump_to_yaml(fname,out_dict)

    
def pd_standardlise(df:pd.DataFrame)->pd.DataFrame:
    return (df-df.values.mean())/df.values.std()

def pd_normalise(df:pd.DataFrame)->pd.DataFrame:
    if df.values.max()-df.values.min() < 1e-9:
        return None
    return (df-df.values.min())/(df.values.max()-df.values.min())


def Plot_sigma_distribution(root,db,solver,basis,outfile_name):
    dataset=AimG_Dataset_RAM(root,db,solver,basis,imag_only=True)
    g_iw_df = pd.DataFrame(np.array([dataset.G[i] for i in range(len(dataset))]))
    dim_G = g_iw_df.shape[1]
    del dataset

    fig = plt.figure(figsize=(8, 9),dpi=300)
    gs = fig.add_gridspec(3, 1, wspace=0,hspace=0,height_ratios=[1,1,1])    
    ax=gs.subplots(sharey=False)

    fontsize= 20

    sns.violinplot(ax=ax[2],data=pd_standardlise(g_iw_df).iloc[:,0:8])
    # ax[2].set_ylabel('Standardlised Imag $G (i\omega_n)$',fontsize=fontsize)
    ax[2].set_ylabel('Z(Imag $G (i\omega_n)$)',fontsize=fontsize)

    sns.violinplot(ax=ax[1],data=pd_normalise(g_iw_df).iloc[:,0:8])
    # ax[1].set_ylabel('Normalised Imag $G (i\omega_n)$',fontsize=fontsize)
    ax[1].set_ylabel('N(Imag $G (i\omega_n)$)',fontsize=fontsize)
    ax[1].set_ylim(top=1,bottom=0)


    sns.violinplot(ax=ax[0],data=g_iw_df.iloc[:,0:8])
    # ax[0].set_ylabel('Imag $G (i\omega_n)$',fontsize=fontsize)
    ax[0].set_ylabel('Imag $G (i\omega_n)$',fontsize=fontsize)
    ax[0].set_ylim(top=0)
    ax[2].set_xlabel('$i\omega_n$',fontsize=fontsize)


    for ax_i in ax:
        ax_i.tick_params(axis='both', which='major', labelsize=fontsize)
        ax_i.tick_params(axis='both', which='minor', labelsize=fontsize)
        ax_i.spines[['top','left','right','bottom']].set_color('black')

    fig.tight_layout()
    fig.savefig(outfile_name)

def Plot_sigma_distribution_old(root,db,solver,basis,outfile_name):
    dataset=AimG_Dataset_RAM(root,db,solver,basis,imag_only=True)

    dim_G = len(dataset.G[0])
    num_sample = len(dataset)
    giw = np.array([dataset.G[i] for i in range(len(dataset))])

    m,low,high = np.zeros(dim_G),np.zeros(dim_G),np.zeros(dim_G)
    for idx in range(dim_G): m[idx],low[idx],high[idx]=mean_confidence_interval(giw[:,idx])
    mean_giw = np.mean(giw,axis=0)
    std_giw = np.std(giw,axis=0)
    del dataset

    fig, ax = plt.subplots(2,1,figsize=(8,6),dpi=600)

    ax[0].errorbar(np.arange(dim_G),mean_giw,std_giw,label='STD:',capsize=3)
    ax[0].errorbar(np.arange(dim_G),m,[low,high],label='$95\%$ CI: ',capsize=3)
    ax[1].boxplot(giw,positions=np.arange(dim_G))
    ax[0].set_ylabel(f'Imag ${basis}(i\omega_n)$(eV)')
    ax[0].legend()

    # ax.hist(giw[:,0],bins=50)
    fig.tight_layout()
    fig.savefig(outfile_name)

def get_all_distribution(data_base_info):
   
    data_distribution_list=[]
    for item in data_base_info:
        data_distribution={}
        
        data_distribution['beta']=item['beta']
        data_distribution['n_imp']=item['n_imp']
        data_distribution['Z']=item['Z']
        data_distribution['U']=item['U']
        data_distribution['W']=item['W']
        data_distribution['eps']=item['eps']

        for item2 in item:
            aim_params= item[item2]
            if type(aim_params) != float and type(aim_params) != int:
                for idx,item3 in enumerate(aim_params):
                    label=' '.join([item2,str(idx+1)])
                    data_distribution[label]=item3
        data_distribution_list.append(data_distribution)
    del data_base_info
    aim_pd_data=pd.DataFrame(data_distribution_list)
    print(f"{aim_pd_data['n_imp'].min()=} {aim_pd_data['n_imp'].max()=}")
    print(f"{aim_pd_data['Z'].min()=} {aim_pd_data['Z'].max()=}")


    ax=aim_pd_data.hist(bins=50,
        xlabelsize=10, ylabelsize=10,
        figsize=(12,12))

    fig = ax[0][0].get_figure()
    print('saving database_viz_full.png')
    # ax[0][1].set_xlim(0., 1)
    ax[0][2].set_xlim(0., 1)
    fig.savefig("database_viz_full.png", format = "png")
    return


def Plot_params_distribution_out(data_base_info):
    E_k_list,V_k_list = [],[]

    for item in data_base_info:
        E_k_list.extend(item['E_p'])
        V_k_list.extend(item['V_p'])

    aim_pd_data=pd.DataFrame(data_base_info)[['n_imp','Z','U','eps']]
    del data_base_info

    half_filled=False
    if aim_pd_data['n_imp'].max()-aim_pd_data['n_imp'].min() < 0.1:   
        print("half_fill activated")
        half_filled=True

    print(f"{aim_pd_data['n_imp'].min()=} {aim_pd_data['n_imp'].max()=}")
    print(f"{aim_pd_data['Z'].min()=} {aim_pd_data['Z'].max()=}")

    fig,ax = plt.subplots(3,2,
                    figsize=(4,3*2),
                    dpi=300,
                    )
    # print(aim_pd_data['n_imp'])
    print(f"plotted EPS")
    sns.histplot(ax=ax[0][1],data=aim_pd_data,x="eps",kde=True,bins=50)
    ax[0][1].set_xlabel(r'$\epsilon_d$ (eV)')
    ax[0][1].set_ylabel(None)

    print(f"plotted U")
    sns.histplot(ax=ax[0][0],data=aim_pd_data,x="U",kde=True,bins=np.linspace(0,10,50))
    ax[0][0].set_xlabel('U (eV)')
    ax[0][0].set_ylabel(None)
    ax[0][0].set_xlim(0,10)

    print(f"plotted EK")
    elim=np.round(np.max(np.abs(E_k_list)))
    if elim == 0: elim=np.round(np.max(np.abs(V_k_list)) / 0.05) * 0.05
    print(elim)
    sns.histplot(ax=ax[1][0],data=E_k_list,kde=True,bins=np.linspace(-elim,elim,50))
    ax[1][0].set_xlabel(r'$\epsilon_k$ (eV)')
    ax[1][0].set_xlim(-elim,elim)

    print(f"plotted VK")
    elim=np.round(np.max(np.abs(V_k_list)))
    if elim == 0: elim=np.round(np.max(np.abs(V_k_list)) / 0.05) * 0.05
    print(elim)
    sns.histplot(ax=ax[1][1],data=V_k_list,kde=True,bins=np.linspace(-elim,elim,50))
    ax[1][1].set_xlabel(r'$V_k$ (eV)')
    ax[1][1].set_ylabel(None)
    ax[1][1].set_xlim(-elim,elim)

    print(f"plotted n_imp")
    color=sns.color_palette("tab10")[0]
    if half_filled:
        ax[2][0].bar(0.5,len(aim_pd_data),0.05,color=color,alpha=0.7,edgecolor='black',linewidth=0.5)
    else:
        sns.histplot(ax=ax[2][0],data=aim_pd_data,x="n_imp",kde=True,bins=np.linspace(0,1,50))
    ax[2][0].set_xlim(0,1)
    ax[2][0].set_xlabel(r'$n$')
    ax[2][0].set_ylabel(None)
    # ax[2][0].set_ylim(0,1)

    print(f"plotted Z")
    sns.histplot(ax=ax[2][1],data=aim_pd_data,x="Z",kde=True,color=color,bins=np.linspace(0,1,50))
    ax[2][1].set_xlabel('Z')
    ax[2][1].set_ylabel(None)
    ax[2][1].set_xlim(0,1)

    for ax2 in ax:
        for _,ax_i in enumerate(ax2):
            ax_i.spines[['top','left','right','bottom']].set_color('black')
    
    plt.tight_layout()
    print(f"saving ./database_viz.png")
    plt.savefig('./database_viz.png')
    return