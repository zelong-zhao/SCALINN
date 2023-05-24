import os,glob,re
import pandas as pd
import contextlib
import torch
import os

def rename_file(fullpath):
    # Check if file exists
    if os.path.isfile(fullpath):
        # Extract the filename from the full path
        dirname, basename = os.path.split(fullpath)
        # Rename the file by prepending 'old_' to the name
        newname = os.path.join(dirname, 'old_' + basename)
        print(f"rename {fullpath} to {newname}")
        os.rename(fullpath, newname)

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


def checkpoints2csv(dir_predix):
    check_dir=os.path.join(dir_predix,'./model_parameter/train/')
    with contextlib.redirect_stdout(None):
        files_list_dict=search_checkpoints(check_dir,'model_epoch_*.pth')
        key_list=exame_keys_checkpoint(files_list_dict[0]['path'])

    out_dict_list=[]
    for item in files_list_dict:
        path=item['path']
        checkpoint = torch.load(path,map_location ='cpu')
        temp_dict={}
        for key in key_list:
            temp_dict[key]=checkpoint[key]
        temp_dict['lr']=checkpoint['scheduler']['_last_lr'][0]
        out_dict_list.append(temp_dict)

        target_path,name=os.path.split(path)[:-1][0],os.path.split(path)[-1].split('.')[0]
        csv_path=os.path.join(target_path,f"{name}.csv")

        csv_path=os.path.join(target_path,f"{name}.csv")
        df=pd.DataFrame([temp_dict])
        df.to_csv(csv_path,index=False)


    df=pd.DataFrame(out_dict_list)
    df_outpath=os.path.join(target_path,'train_val_loss.csv')
    print(f"saving to {df_outpath=}")
    if os.path.exists(df_outpath):
        print(f"{df_outpath} exist")
        print(pd.read_csv(df_outpath))
        rename_file(df_outpath)
    df.to_csv(df_outpath,index=False)
    print(f"after updated")
    print(pd.read_csv(df_outpath))


def exame_keys_checkpoint_csv(path_checkpoint):
    checkpoint = pd.read_csv(path_checkpoint)
    key_list=[]
    check_key_list=['train','test','epoch']
    for key in checkpoint.head():
        for key2 in check_key_list:
            if key2 in key.lower():
                key_list.append(key)
    print(f"find {key_list} in checkpoints \n")
    return key_list

def csv_checkpoints2csv(dir_predix):
    check_dir=os.path.join(dir_predix,'./model_parameter/train/')
    with contextlib.redirect_stdout(None):
        files_list_dict=search_checkpoints(check_dir,'model_epoch_*.csv')
        assert files_list_dict is not None, 'did not find any model_epoch_*.csv'
        key_list=list(pd.read_csv(files_list_dict[0]['path']).columns)
    
    print(f"merge small csv to large: {key_list=}")

    out_dict_list=[]
    for item in files_list_dict:
        path=item['path']
        checkpoint = pd.read_csv(path)
        temp_dict={}
        for key in key_list:
            temp_dict[key]=checkpoint.loc[0,key]

        out_dict_list.append(temp_dict)

    target_path,_=os.path.split(path)[:-1][0],os.path.split(path)[-1].split('.')[0]

    df=pd.DataFrame(out_dict_list)
    df_outpath=os.path.join(target_path,'train_val_loss.csv')
    print(f"saving to {df_outpath=}")
    if os.path.exists(df_outpath):
        print(f"{df_outpath} exist")
        print(pd.read_csv(df_outpath))
        rename_file(df_outpath)
    df.to_csv(df_outpath,index=False)
    print(f"after updated")
    print(pd.read_csv(df_outpath))