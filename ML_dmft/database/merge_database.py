from shutil import rmtree
import numpy as np
import os.path
import sys
import pickle #dump the data
import argparse,shutil
import pandas as pd
from time import perf_counter
from ML_dmft.torch_model.checkpoints_tools import search_checkpoints
import contextlib

def merge_csv():
    csv_list=[]
    pwd=os.getcwd()

    print('only check db_1 db_2 ... up to db_100')
    db_list = [ os.path.join(pwd,'db_%d'%i) for i in range(1,100) if os.path.isdir(os.path.join(pwd,'db_%d'%i))]

    for dir_name in db_list:
        db_dir = os.path.join(dir_name,'db')
        if os.path.isdir(db_dir):
            csv_list.extend([ os.path.join(db_dir,item) for item in os.listdir(db_dir) if item.endswith('.csv') ])
        else:
            print(f'{db_dir} is not a db!!!')

    csv_sep=[]
    for item in csv_list:
        csv_sep.append(item.split(os.sep)[-1])
    csv_sep=list(set(csv_sep))

    csv_path_dict={}
    for csv_file in csv_sep:
        one_type=[]
        for abs_path in csv_list:
            if abs_path.split(os.sep)[-1] == csv_file:
                one_type.append(abs_path)
        csv_path_dict[csv_file]=one_type

    target_dir=os.path.join(pwd,'merged_db')
    if os.path.isdir(target_dir):
        rmtree(target_dir)
    
    os.makedirs(target_dir)
    print('create',target_dir)

    for key in csv_path_dict:
        out_file=os.path.join(pwd,'merged_db',key)
        # print(f'{out_file} created')
        for csv_in in csv_path_dict[key]:
            print('reading ',csv_in)
            df = pd.read_csv(csv_in)
            df.to_csv(out_file, mode="a", index=False)         
            
def merge_db():
    prefix_database=os.getcwd()
    target_out='./'
    db_list = [ item for item in os.listdir('./') if os.path.isdir(os.path.join(item)) ]
    load_database(db_list,target_out,prefix_database)

def splits_list(input_list,num_db,whoami):
    index_list,item_list=[],[]
    for idx,item in enumerate(input_list):
        if idx%num_db!=whoami:continue
        index_list.append(idx)
        item_list.append(item)
    return index_list,item_list

def splites_bigdb_to_small(num_db_2_split):
    # num_db_2_split = 32
    pwd=os.getcwd()

    # check if ./db_1 exist
    for i in range(1,num_db_2_split+1):
        if os.path.isdir(os.path.join(pwd,'db_%d'%i)):
            print(f"{os.path.join(pwd,'db_%d'%i)} exit")
            raise FileExistsError('File already exist')
    
    in_file=os.path.join(pwd,'db','aim_params.csv')
    print(f"Reading: {in_file}")
    df = pd.read_csv(in_file,header=None)
    num_data = len(df)
    print(f"num of data: {num_data}")

    db_to_index_list=[]
    for which_db in range(num_db_2_split):
        index_list=[]
        for idx in range(num_data):
            if idx%num_db_2_split!=which_db:continue
            index_list.append(idx)
        db_to_index_list.append(index_list)
        print(f"db_{which_db+1} has {len(db_to_index_list[which_db])} samples")
    

    for which_db in range(num_db_2_split):

        sub_db_df = df.take(db_to_index_list[which_db])
        db_index = which_db+1
        sub_db_path=os.path.join(pwd,'db_%d'%db_index,'db')

        os.makedirs(sub_db_path)
        print(f"makdirs {sub_db_path}, write to aim_params.csv")
        out_file = os.path.join(sub_db_path,'aim_params.csv')
        sub_db_df.to_csv(out_file, mode="w", index=False,header=None)



def merge_star_csv():
    pwd=os.getcwd()
    db_list = [ os.path.join(pwd,'db_%d'%i) for i in range(1,100) if os.path.isdir(os.path.join(pwd,'db_%d'%i))]
    with contextlib.redirect_stdout(None):
        aim_csv_star_list = search_checkpoints(os.path.join(db_list[0],'db'),"*_aim_params.csv")
    if aim_csv_star_list is not None: 
        print(aim_csv_star_list)
        print(f'aim_csv_star exists so search it {aim_csv_star_list}')
        for searched_file in aim_csv_star_list:
            aim_csv_list2 = []
            star_aim_csv_name = os.path.split(searched_file['path'])[-1]
            out_file=os.path.join(pwd,'merged_db',star_aim_csv_name)

            print(f"{star_aim_csv_name}")
            for dir_name in db_list:
                db_dir = os.path.join(dir_name,'db')
                if os.path.isdir(db_dir):
                    aim_csv_list2.append(os.path.join(dir_name,'db',star_aim_csv_name))
                else:
                    print(f'{db_dir} is not a db!!!')
                    raise FileExistsError(f'Stop as {db_dir=} doest exist.')

        # number of file list
            for csv_in in aim_csv_list2:
                # print('reading ',csv_in)
                df = pd.read_csv(csv_in,header=None)
                df.to_csv(out_file, mode="a", index=False,header=None)
                # print(df)
            # print(f'write to {out_file}')


def merge_small_db():
    aim_csv_list = []
    pwd=os.getcwd()

    target_dir=os.path.join(pwd,'merged_db')
    if os.path.isdir(target_dir):
        rmtree(target_dir)
    
    os.makedirs(target_dir)
    print('create',target_dir)

    merge_star_csv()


    print('only check db_1 db_2 ... up to db_100')
    db_list = [ os.path.join(pwd,'db_%d'%i) for i in range(1,10000) if os.path.isdir(os.path.join(pwd,'db_%d'%i))]

    for dir_name in db_list:
        db_dir = os.path.join(dir_name,'db')
        if os.path.isdir(db_dir):
            aim_csv_list.extend([ os.path.join(db_dir,item) for item in os.listdir(db_dir) if item == 'aim_params.csv' ])
        else:
            print(f'{db_dir} is not a db!!!')
            raise FileExistsError(f'Stop as {db_dir=} doest exist.')

    # save csv to there
    # number of file list
    num_file=[]
    out_file=os.path.join(pwd,'merged_db','aim_params.csv')
    for csv_in in aim_csv_list:
        print('reading ',csv_in)
        df = pd.read_csv(csv_in,header=None)
        df.to_csv(out_file, mode="a", index=False,header=None)
        num_file.append(len(df))
        print(df)
    print(f'write to {out_file}')


    solver_list=[]
    dir_name=db_list[0]
    db_dir = os.path.join(dir_name,'db')
    for solver in os.listdir(db_dir):
        if os.path.isdir(os.path.join(dir_name,'db',solver)):  
            solver_list.append(solver)
    print(f'\n solvers find in db_1 {solver_list} \n')
    for solver in solver_list:
        os.makedirs(os.path.join(pwd,'merged_db',solver))
        print(f"mkdir {os.path.join(pwd,'merged_db',solver)}")

    #mv ./db_1/db/solver/0 ./merge_db/solver/0
    for solver in solver_list:
        counter=0
        for idx,dir_name in enumerate(db_list):
            db_dir = os.path.join(dir_name,'db')
            solver_path=os.path.join(dir_name,'db',solver)
            if not os.path.isdir(solver_path):
                raise ValueError(f"!! solver not find{solver_path}")
            else:
                number_of_dir=len(os.listdir(solver_path))
                if number_of_dir != num_file[idx]:
                    raise ValueError(f'{solver_path} number of dirs and aim not match')
                for dat_idx in range(number_of_dir):
                    old_data_path=os.path.join(solver_path,str(dat_idx))
                    new_data_path=os.path.join(os.path.join(pwd,'merged_db',solver,str(counter)))
                    if not os.path.isdir(old_data_path):
                        raise ValueError(f"{old_data_path} does not exist!")
                    print(f"move {old_data_path} to {new_data_path}")
                    shutil.move(old_data_path,new_data_path)
                    counter=counter+1

def merge_small_to_one():
    """
    FILES=./db/ED_KCL_4/0/G_tau.dat
    """

    pwd=os.getcwd()
    db_dir=os.path.join(pwd,'db')
    csv_in=os.path.join(pwd,'db','aim_params.csv')
    aim_csv=pd.read_csv(csv_in,header=None)

    searched_key='G_tau'
    print(f'only look at {searched_key}')
    solver_list=[]
    for solver in os.listdir(db_dir):
        if os.path.isdir(os.path.join(db_dir,solver)):  
            solver_list.append(solver)
    for solver in solver_list:
        solver_path=os.path.join(db_dir,solver)
        number_of_dir=len(os.listdir(solver_path))

        if number_of_dir != len(aim_csv):
                raise ValueError(f'num in aim_params.csv and solver dir are different')
        else:
            print(f' number of elements:{number_of_dir}')
        Target_combined_path=os.path.join(db_dir,f"{solver}_{searched_key}.csv")
        find_del_file(Target_combined_path)
        start_time=perf_counter()
        for dat_idx in range(number_of_dir):
            Sing_G_path=os.path.join(solver_path,str(dat_idx),f'{searched_key}.dat')
            G=accessG(Sing_G_path)
            writeG(Target_combined_path,G)
        print(f"time finished {perf_counter()-start_time}")

def find_del_file(path):
    if os.path.isfile(path):
        os.remove(path)
        print(f'{"delete":10.5} {path}')
    print(f'{"writting to":10.5} {path}')

def accessG(G_path):
    G=[]
    with open(G_path,'r') as F:
        for lines in F:
            G.append(float(lines))
    G=np.array(G).reshape((1,len(G)))
    return G
def writeG(G_path,G):
    with open(G_path,'a') as F:
        # G_out=' '.join([str(item) for item in G[0]])
        # F.write(G_out)
        # F.write('\n')
        np.savetxt(F,G)

def main():
    parser = argparse.ArgumentParser(description='merge database')
    parser.add_argument('--merge-csv', action='store_true', default=False,
                    help='merge-csv')
    parser.add_argument('--merge-pickle', action='store_true', default=False,
                help='merge pickle dictionary')
    parser.add_argument('--merge-small-db', action='store_true', default=False,
                help='merge smalls')
    parser.add_argument('--merge-small-to-one', action='store_true', default=False,
                help='merge files in ./db/solver/1/G_tau.csv')

    parser.add_argument('--splits_db_to_small', action='store_true', default=False,
                help='splits big db to small one')

    parser.add_argument('--num_db',type=int,help='num of db (int)',required='--splits_db_to_small' in sys.argv)

    args = parser.parse_args()

    if args.merge_pickle:
        merge_db()
    
    if args.merge_csv:
        merge_csv()

    if args.merge_small_db:
        merge_small_db()
    
    if args.merge_small_to_one:
        merge_small_to_one()

    if args.splits_db_to_small:
        splites_bigdb_to_small(args.num_db)