from ML_dmft.database.dataset import AIM_Dataset_Rich,AIM_Dataset
from ML_dmft.database.database_plot import plot_tool
from ML_dmft.database.dataset_tool import get_all_distribution,Plot_sigma_distribution,Plot_params_distribution_out,meansure_stats,meansure_aim_param_stats
from ML_dmft.triqs_interface.v2 import triqs_plot_variation
import os,sys
import numpy as np
import argparse

def read_args():
    parser = argparse.ArgumentParser(description='./db/solver/ analysis tools')
    parser.add_argument('--db', type=str,default='./',
                        help='db to test default(./)')
    parser.add_argument('--root-database', type=str, default=os.getcwd(),
                    help='root to the dir of database default(pwd)')

    parser.add_argument('--summary',action='store_true', default=False,
                    help='(action) Find AIM distribution default(False)')

    parser.add_argument('--info',action='store_true', default=False,
                    help='(action) calculate statistic information default(False)')


    parser.add_argument('--one_solver',action='store_true', default=False,
                    help='(action) one_solver default(False) --index and solver to control')

    parser.add_argument('--one_solver_all_method',action='store_true', default=False,
                    help='(action) one_solver_all_method default(False) --index and solver to control')
    
    parser.add_argument('--index', type=int, default=0,
                    help='index to meansure')
    parser.add_argument('--solver', type=str,required='--summary' in sys.argv,
                    help='solver for plotting one solver all method')
    parser.add_argument('--basis', type=str,required='--info' in sys.argv,
                    help='basis for plotting one solver all method')
        
    parser.add_argument('--plot_two_solver_two_method',action='store_true', 
                        default=False,
                        help='(action) plot_two_solver_two_method default(False) --index, solver,solver2 G_method G_method2 Sigma_method Sigma_method2 to control')

    parser.add_argument('--solver2', type=str,
                    help='solver to meansure for compare two solver')
    parser.add_argument('--G_method', type=int, default=0,choices=[0,1,2],
                    help='G_method')
    parser.add_argument('--Sigma_method', type=int, default=0,choices=[0,1,2],
                    help='Sigma method')

    parser.add_argument('--G_method2', type=int, default=0,choices=[0,1,2],
                    help='G_method2')
    parser.add_argument('--Sigma_method2', type=int, default=0,choices=[0,1,2],
                    help='Sigma method2')

    args = parser.parse_args()
    return args

def summary(args):
    """
    Plot database distribution
    """
    dataset=AIM_Dataset(root=args.root_database,
                        db=args.db,
                        solver=args.solver,
                        load_aim_csv_cpu=False
                        )
    data_dict=[]
    for idx in range(len(dataset)):
        data_dict.append(dataset(idx,'aim_params'))

    for idx in range(len(dataset)):
        data_dict[idx]['beta']=dataset(idx,'beta')
        data_dict[idx]['Z']=dataset(idx,'Z')
        data_dict[idx]['n_imp']=dataset(idx,'n_imp')
    
    # get_all_distribution(data_dict)
    Plot_params_distribution_out(data_dict)


def plot_distribution(args):
    meansure_stats(args.root_database,args.db,args.solver,args.basis)
    meansure_aim_param_stats(args.root_database,args.db,args.solver)
    Plot_sigma_distribution(args.root_database,args.db,args.solver,args.basis,f"{args.solver}_{args.basis}_distribution.png")


def plot_one_solver_all_method(args):
    dataset=AIM_Dataset_Rich(root=args.root_database,
                        db=args.db,
                        solver=args.solver,
                        index=args.index)
    a=triqs_plot_variation(dataset)
    a.plot_one_solver_all_method()

def plot_one_solver(args):
    plot=plot_tool(root=args.root_database,
                        db=args.db,
                        solver=args.solver,
                        index=args.index)
    plot.plot_iw_obj()


def plot_two_solver(args):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use(plt.style.available[-2])
    x_window=10

    dataset_solver1=AIM_Dataset_Rich(root=args.root_database,
                        db=args.db,
                        solver=args.solver,
                        index=args.index)
    
    solver1=triqs_plot_variation(dataset_solver1)  
    solver1.proceed(args.G_method,args.Sigma_method)

    dataset_solver2=AIM_Dataset_Rich(root=args.root_database,
                        db=args.db,
                        solver=args.solver2,
                        index=args.index)
    solver2=triqs_plot_variation(dataset_solver2)  
    solver2.proceed(args.G_method2,args.Sigma_method2)
    

    print('Z GT and Density GT is from solver,G_method,Sigma_method')
    Z_GT=solver1.Z
    density_GT=solver1.density

    fig, ax = plt.subplots(3, 4,figsize=(16,9),dpi=300)

    solver1.plot_all_info(ax,ls='',
                    marker='.',
                    plot_ground_truth=True,
                    xwindow=x_window,
                    legend=solver1.legend,
                    Z_GT=Z_GT,
                    density_GT=density_GT,
                    )

    solver2.plot_all_info(ax,ls='',
                    marker='.',
                    plot_ground_truth=False,
                    xwindow=x_window,
                    legend=solver2.legend,
                    Z_GT=Z_GT,
                    density_GT=density_GT,
                    )

    fig.suptitle(solver1.title,fontsize=10)
    fig.tight_layout()
    print(f'ALL_GF_{args.solver}_{args.solver2}.png is created')
    fig.savefig(f'ALL_GF_{args.solver}_{args.solver2}.png')


def main():
    args=read_args()
    if args.summary:
        summary(args)
    if args.info:
        plot_distribution(args)
    if args.one_solver:
        plot_one_solver(args)
    if args.one_solver_all_method:
        plot_one_solver_all_method(args)
    if args.plot_two_solver_two_method:
        plot_two_solver(args)    
