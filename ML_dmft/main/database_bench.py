from ML_dmft.database.benchmark_dataset import Benchmark_Procced,Plot_Benchmark
from ML_dmft.utility.mpi_tools import mpi_rank,mpi_size
import argparse,os,warnings,sys

def read_args():
    parser = argparse.ArgumentParser(description='./db/solver/ analysis tools')
    parser.add_argument('--db', type=str,default='./',
                        help='db to test default(./)')
    parser.add_argument('--root-database', type=str, default=os.getcwd(),
                    help='root to the dir of database default(pwd)')

    parser.add_argument('--procceed-benchmark','-S',action='store_true', default=False,
                    help='(action) Proceed benchmark calculation(mpi)')

    parser.add_argument('--plot-benchmark','-P',action='store_true', default=False,
                    help='(action) Proceed benchmark calculation(mpi)')
                    
    parser.add_argument('--index', type=int, default=0,
                    help='index to meansure')
    parser.add_argument('--solver', type=str,
                    help='solver for plotting one solver all method',required=True)
        
    parser.add_argument('--solver2', type=str,
                    help='solver to meansure for compare two solver',required=True)
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

def plot_benchmark(args):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use(plt.style.available[-2])

    fig, ax = plt.subplots(2, 4,figsize=(16,6),dpi=300)
    plot_tool=Plot_Benchmark(root=args.root_database,
                             db=args.db,
                             ground_truth_solver=args.solver,
                             solver2test=args.solver2)

    plot_tool.plot_Z(ax[1,2])
    plot_tool.plot_n(ax[1,3])
    plot_tool.plot_G_tau(ax[0,0])
    plot_tool.plot_G_l(ax[0,1])

    plot_tool.plot_G_iw(ax[0,2],ax[0,3])
    plot_tool.plot_Sigma_iw(ax[1,0],ax[1,1])

    fig.tight_layout()
    fig.savefig('benchmark.png')
    print('saving benchmark.png')

def main():
    args=read_args()
    if args.procceed_benchmark:
        input_dict=dict(root=args.root_database,
                        db=args.db,
                        solver1=args.solver,
                        solver2=args.solver2,
                        G_method1=args.G_method,
                        G_method2=args.G_method2,
                        Sigma_method1=args.Sigma_method,
                        Sigma_method2=args.Sigma_method2)
                        
        bp=Benchmark_Procced(**input_dict)

    elif args.plot_benchmark:
        if not mpi_size()==1: raise SystemError('mpi is not allowed for plotting')
        plot_benchmark(args)
    else:
        if mpi_rank()==0: warnings.warn('no option selected',SyntaxWarning)