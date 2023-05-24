
from ML_dmft.database.dataset import AIM_Dataset_Rich,AIM_Dataset,AIM_dataset_meshG
from ML_dmft.triqs_interface.v2 import triqs_plot_variation
import argparse
import numpy as np
import os

# Goal write to db A_w by index
def get_args():
    parser = argparse.ArgumentParser(description='Tools to generate A(w)')
    parser.add_argument('--root', type=str, default='./', metavar='f',
                    help='root to the dir of database')
    parser.add_argument('--db', type=str, default='./', metavar='f',
                help='name of the database default(./)')
    parser.add_argument('--solver', type=str,required=True,
                        help='solver to learn. root database')
    parser.add_argument('--index', type=int,default=None)
    parser.add_argument('--xlim_left', type=float,default=-8.)
    parser.add_argument('--xlim_right', type=float,default=8.)
    parser.add_argument('--num_points', type=int,default=200)
    parser.add_argument('--freq_offset', type=float,default=None)
    args = parser.parse_args()
    return args


def build_real_feq(root:str,db:str,solver:str,index:int,
                   num_points=1024,
                   xlim_left=-8,
                   xlim_right=8,
                   freq_offset:float=None):
    """
    write to root/db/solver/index/Aw.dat
    """
    target_path=os.path.join(root,db,'db',solver,str(index),'A_w.dat')

    print(f"saving to {target_path}")
    dataset_solver1=AIM_Dataset_Rich(root=root,
                    db=db,
                    solver=solver,
                    index=index)
    
    solver1=triqs_plot_variation(dataset_solver1)  
    solver1.proceed(1,1)
    mesh,A_w=solver1.density_of_states(
                    Matsu_points=num_points,
                    xlim_left=xlim_left,
                    xlim_right=xlim_right,
                    freq_offset=freq_offset)

    _n,_=mesh.shape
    Data_input=np.zeros((_n,2))
    Data_input[:,0],Data_input[:,1]=mesh[:,0],A_w[:,0]

    np.savetxt(target_path,Data_input,fmt='%20.10e \t %20.10e',delimiter='\t')


def main():
    args=get_args()

    db=AIM_Dataset(args.root,args.db,args.solver)

    if args.index is None:
        print('will proccess for all index')
    
    for index in range(len(db)):
        if args.index is not None:
            index = args.index

        build_real_feq(root=args.root,
                    db=args.db,
                    solver=args.solver,
                    index=index,
                    num_points=args.num_points,
                    xlim_left=args.xlim_left,
                    xlim_right=args.xlim_right,
                    freq_offset=args.freq_offset)
        
        if args.index is not None: 
            break
            

if __name__=='__main__':
    build_real_feq('./','./','ED_KCL_7',0)