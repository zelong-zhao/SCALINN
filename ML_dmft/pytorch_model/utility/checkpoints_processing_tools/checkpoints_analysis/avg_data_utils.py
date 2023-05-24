import os,warnings
from ...train_tools.pth2csv import csv_checkpoints2csv
import copy
import pandas as pd
import numpy as np

def ls_dir(dir_path:str)->list:
    # Get a list of all items in the directory
    items = os.listdir(dir_path)

    # Loop through the items and check if each one is a directory
    found_dir=[]
    for item in items:
        task_path = os.path.join(dir_path, item)
        if os.path.isdir(task_path):
            model_param_path=os.path.join(task_path,'model_parameter/train/')
            train_val_csv_path=os.path.join(task_path,'model_parameter/train/','train_val_loss.csv')

            if  os.path.isdir(model_param_path):

                if os.path.isfile(train_val_csv_path):
                    found_dir.append(item)
                else:
                    try:
                        csv_checkpoints2csv(task_path)
                    except:
                        print(f"{model_param_path} exists but not checkpoints foud!!")
                    else:
                        if os.path.isfile(train_val_csv_path):
                            found_dir.append(item)
            else:
                print(f"!!!{task_path} exists but {train_val_csv_path} does not exists!!!!")
    return found_dir

def check_dir_have_train_val_csv(source_dir:list,task_dirs:list):
    out_task_dirs=copy.deepcopy(task_dirs)

    for _dir in source_dir:
        for task in out_task_dirs:
            task_path=os.path.join(_dir,task)
            model_param_path=os.path.join(task_path,'model_parameter/train/')
            train_val_csv_path=os.path.join(_dir,task,'model_parameter/train/','train_val_loss.csv')
            if  os.path.isdir(model_param_path):
                if not os.path.isfile(train_val_csv_path):
                    try:
                        csv_checkpoints2csv(task_path)
                    except:
                        print(f"failed to mk checkpoints")
                        out_task_dirs.remove(task)
            else:
                out_task_dirs.remove(task)
                warnings.warn(f"{model_param_path} does not exit")    
    if task_dirs!=out_task_dirs:
        for item in task_dirs:
            if item not in out_task_dirs:
                print(f'!!!{item} removed from task as not in other RUN')
    for item in out_task_dirs:
        print(f"checked {item=} can be averaged")
    return out_task_dirs

def exame_keys_checkpoint_csv(path_checkpoint):
    checkpoint = pd.read_csv(path_checkpoint)
    key_list=[]
    check_key_list=['train','test']
    for key in checkpoint.head():
        for key2 in check_key_list:
            if key2 in key.lower():
                key_list.append(key)
    print(f"find {key_list} in checkpoints \n")
    return key_list

def average_csv_columns(file_paths, output_file, columns_to_collect):

    min_len=np.inf
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        min_len=min(len(df),min_len)
        # print(f"{file_path=}\n{df}")

    stacked = None
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        if len(df) != min_len:
            print(f"! {len(df)=} but {min_len=}")
        if stacked is None:
            stacked = df[:min_len]
        else:
            stacked = pd.concat([stacked, df[:min_len]])

    average=pd.DataFrame(index=list(set(stacked.index.tolist())),
                         columns=stacked.columns.to_list()
                         )
    # print(stacked.index.tolist())
    # print(average)
    for idx in list( set(stacked.index.tolist()) ):
        average.loc[idx,['epoch']]=stacked.loc[idx,'epoch'].values[0]
        average.loc[idx,['lr']]=stacked.loc[idx,'lr'].values[0]
        average.loc[idx,columns_to_collect]=stacked.loc[idx][columns_to_collect].mean()

    print(f"{average=}")
    average.to_csv(output_file, index=False)
    return average

def average_different_run(source_dir:list,task_dirs:list,OUT_DIR:str):
    train_test_keys=exame_keys_checkpoint_csv(os.path.join(source_dir[0],task_dirs[0],'model_parameter/train/','train_val_loss.csv'))
    print(f"{train_test_keys=}")
    for task in task_dirs:
        tem_path=[]
        for _dir in source_dir:
            task_path=os.path.join(_dir,task)
            model_param_path=os.path.join(task_path,'model_parameter/train/')
            train_val_csv_path=os.path.join(_dir,task,'model_parameter/train/','train_val_loss.csv')
            assert os.path.isfile(train_val_csv_path)
            tem_path.append(train_val_csv_path)
        print(f"{task=} reading {tem_path=}")
        outpath=os.path.join(OUT_DIR,task,'model_parameter/train/')
        os.makedirs(outpath)
        out_file_path=os.path.join(outpath,'train_val_loss.csv')
        average=average_csv_columns(file_paths=tem_path,
                            output_file=out_file_path,
                            columns_to_collect=train_test_keys)
        # print(f"{average=}")
        print(f'write to {out_file_path=}')


def avg_data_dir(source_dir=['./RUN1','./RUN2','./RUN3'],target_dir='Average_RUN'):
    r"""
    READ A LIST that have RUN1 RUN2
    """
    CURRENT_DIR, _ = os.path.split(source_dir[0])
    OUT_DIR = os.path.join(CURRENT_DIR,target_dir)
    print(f"In avg_data_dir: {CURRENT_DIR=}")    
    for item in source_dir:
        assert os.path.isdir(item),'source dir did not find RUN1 RUN2'
    assert not os.path.isdir(OUT_DIR),'target dir should not exits'

    task_dirs=ls_dir(source_dir[0])
    assert task_dirs != [],'source dir should not be empty'
    task_dirs=check_dir_have_train_val_csv(source_dir,copy.deepcopy(task_dirs))
    
    average_different_run(source_dir,task_dirs,OUT_DIR)