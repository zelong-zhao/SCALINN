import os,re,glob
import pandas as pd
import torch
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use(plt.style.available[-2])

from ..train_tools.pth2csv import checkpoints2csv,csv_checkpoints2csv
from .evalutae_tools import plot_df_adv


def search_checkpoints(dir,rule):
    """
    given directory and rule to find possible files and rank by then number
    [{"index":index,
     "path":path,
    }]
    """
    print(f"searching {os.path.join(dir,rule)}")
    out_file_list=[]
    path_list=glob.glob(os.path.join(dir,rule))
    for path in path_list:
        file=path.split(os.sep)[-1]
        index=int(re.search('[0-9]+',file).group(0))
        out_file_list.append({"index":index,
                            "path":path,
                            })
    out_file_list=sorted(out_file_list, key=lambda d: d['index']) 
    
    for item in out_file_list:
        print(f" find {item['index']} of {item['path']}")
    print('\n')
    if len(out_file_list) == 0:
        out_file_list=None
    return out_file_list

def exame_keys_checkpoint(path_checkpoint):
    checkpoint = torch.load(path_checkpoint,map_location ='cpu')
    key_list=[]
    check_key_list=['train','test','epoch']
    for key in checkpoint:
        for key2 in check_key_list:
            if key2 in key.lower():
                key_list.append(key)
    print(f"find {key_list} in checkpoints \n")

    return key_list

def make_loss_dict(files_list_dict:list,key_list:list):
    """
    OUT LOSS DICT
    {"train MSE":train MSE
    "}
    """
    out_dict_list=[]
    for item in files_list_dict:
        path=item['path']
        checkpoint = torch.load(path,map_location ='cpu')
        temp_dict={}
        for key in key_list:
            temp_dict[key]=checkpoint[key]
        temp_dict['lr']=checkpoint['scheduler']['_last_lr'][0]
        out_dict_list.append(temp_dict)
    return out_dict_list

def pth2csv(args):
    for item in args.prefix_list:
        checkpoints2csv(item)

def csv2csv(args):
    for item in args.prefix_list:
        csv_checkpoints2csv(item)

def plot_loss_multiple(args):
    fig, ax = plt.subplots(2,2,figsize=(9,6),dpi=300)
    prefix_list=args.prefix_list
   
    for index,item in enumerate(prefix_list):
        check_dir=os.path.join(item,args.directory)
        if args.use_pth_first:
            pth2csv(args)
            files_list_dict=search_checkpoints(check_dir,args.file_rule)
            key_list=exame_keys_checkpoint(files_list_dict[0]['path'])
            out_dict_list=make_loss_dict(files_list_dict,key_list)
            df=pd.DataFrame(out_dict_list)
        else:
            target_path=os.path.join(check_dir,'train_val_loss.csv')
            print(f"reading {target_path}")
            if args.update_csv:
                print(f"updating csv {args.update_csv=}")
                csv2csv(args)
            if not os.path.exists(target_path):
                print(f"{target_path} doesn't exist!!!!! so make it")
                pth2csv(args)
            df=pd.read_csv(target_path)
        
        label=str(item)+' '
        if args.legend_list is not None:
            label=str(args.legend_list[index])+' '

        plot_logscale=args.plot_logscale #IF F
        title=args.title
        color=sns.color_palette("tab10")[index]
        plot_df_adv(ax[0,0],df,dfkey="test MSE",yname="MSE",label='',color=color,ls='-',marker='+',logsclae=plot_logscale)
        plot_df_adv(ax[0,0],df,dfkey="train MSE",yname="MSE",label=label,color=color,ls='',marker='.',logsclae=plot_logscale)

        plot_df_adv(ax[1,0],df,dfkey="test Matsu",yname="Matsu",label='',color=color,ls='-',marker='+',logsclae=plot_logscale)
        plot_df_adv(ax[1,0],df,dfkey="train Matsu",yname="Matsu",label=label,color=color,ls='',marker='.',logsclae=plot_logscale)
    
        plot_df_adv(ax[0,1],df,dfkey="test STD",yname="MSE STD",label='',color=color,ls='-',marker='+',logsclae=plot_logscale)
        plot_df_adv(ax[0,1],df,dfkey="train STD",yname="MSE STD",label=label,color=color,ls='',marker='.',logsclae=plot_logscale)

        plot_df_adv(ax[1,1],df,dfkey="lr",yname="lr",label=label,color=color,ls='-',marker='',logsclae=False)

    # legend1=ax[1].legend()
    # handles, labels = plt.gca().get_legend_handles_labels()
    from matplotlib.lines import Line2D
    line1 = Line2D([0], [0],linestyle='',marker='.', label='train loss', color=sns.color_palette("tab10")[0])
    line2 = Line2D([0], [0],linestyle='',marker='+', label='val loss',color=sns.color_palette("tab10")[0])
    # handles.extend([line1,line2])
    handles=[line1,line2]
    ax[1,0].legend(handles=handles)
    # plt.gca().add_artist(legend1)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig('err_loss.png')


def plot_IPT_vs_ML_multiple(args):
    fig, ax = plt.subplots(2,figsize=(6,6),dpi=300)
    logsclae=True
    prefix_list=list(args.prefix_list)

    for index,item in enumerate(prefix_list):

        check_file=os.path.join(item,args.directory,args.file_rule)
        if not os.path.isfile(check_file):
            raise ValueError(f'{check_file} does not exist')

        if item=='./':
            label=''
        else:
            label=f'{item}'

        color=sns.color_palette("tab10")[index]
        df=pd.read_csv(check_file)
        plot_df_adv(ax[0],df,dfkey="matsu_MSE(ML,TRUTH)",yname="matsu_MSE",color=color,label=label,ls='-',marker='.',logsclae=logsclae)
        plot_df_adv(ax[1],df,dfkey="matsu_MSE(ML,TRUTH)/matsu_MSE(IPT,TRUTH)",color=color,yname="matsu_MSE(ML,TRUTH)/matsu_MSE(IPT,TRUTH)",label='',ls='-',marker='.',logsclae=logsclae)

    plot_df_adv(ax[0],df,dfkey="matsu_MSE(IPT,TRUTH)",yname="matsu_MSE",color='black',label='IPT',ls='-',marker='+',logsclae=logsclae)
    # legend1=ax[1].legend()
    # handles, labels = plt.gca().get_legend_handles_labels()
    from matplotlib.lines import Line2D
    line1 = Line2D([0], [0],linestyle='',marker='.', label='matsu_MSE(ML,TRUTH)', color=sns.color_palette("tab10")[0])
    line2 = Line2D([0], [0],linestyle='',marker='+', label='matsu_MSE(IPT,TRUTH)',color='black')
    # handles.extend([line1,line2])
    handles=[line1,line2]
    ax[1].legend(handles=handles)

    # ax.legend(frameon=True)
    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig('PredictedG_vs_IPT.png')


def plot_IPT_vs_ML_multiple_back(args):
    fig, ax = plt.subplots(2,figsize=(6,6),dpi=300)
    logsclae=True
    prefix_list=list(args.prefix_list)

    for index,item in enumerate(prefix_list):

        check_file=os.path.join(item,args.directory,args.file_rule)
        if not os.path.isfile(check_file):
            raise ValueError(f'{check_file} does not exist')

        if item=='./':
            label=''
        else:
            label=f'{item}'

        color=sns.color_palette("tab10")[index]
        df=pd.read_csv(check_file)

        plot_df_adv(ax[0],df,dfkey="MSE(ML,TRUTH)",yname="MSE",color=color,label=label,ls='-',marker='.',logsclae=logsclae)
        plot_df_adv(ax[1],df,dfkey="MSE(ML,TRUTH)/MSE(IPT,TRUTH)",color=color,yname="MSE(ML,TRUTH)/MSE(IPT,TRUTH)",label='',ls='-',marker='.',logsclae=logsclae)

    plot_df_adv(ax[0],df,dfkey="MSE(IPT,TRUTH)",yname="MSE",color='black',label='IPT',ls='-',marker='+',logsclae=logsclae)
    # legend1=ax[1].legend()
    # handles, labels = plt.gca().get_legend_handles_labels()
    from matplotlib.lines import Line2D
    line1 = Line2D([0], [0],linestyle='',marker='.', label='MSE(ML,TRUTH)', color=sns.color_palette("tab10")[0])
    line2 = Line2D([0], [0],linestyle='',marker='+', label='MSE(IPT,TRUTH)',color='black')
    # handles.extend([line1,line2])
    handles=[line1,line2]
    ax[1].legend(handles=handles)

    # ax.legend(frameon=True)
    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig('PredictedG_vs_IPT.png')




