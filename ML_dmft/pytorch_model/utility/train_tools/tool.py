from typing import Callable
import string
import torch
import glob
import re,os
import numpy as np
from time import perf_counter


def return_dataloader_dim(dataloader):
    loader=iter(dataloader)
    (param,sample_G),_=next(loader)
    dim=param.shape[1]
    time_step=sample_G.shape[1]
    print(f"input dim={dim} time-step={time_step}")
    return dim,time_step

def device_info():
    print('\n'+50*'#')
    print(f'is the device aviliable {torch.cuda.is_available()}')
    print(f'number of device {torch.cuda.device_count()}')
    print(f'name of device {torch.cuda.get_device_name(0)}')
    print(50*'#','\n')

    num_gpu=torch.cuda.device_count()
    return num_gpu

def dataset_splite(args,kargs,Train_dataset):
    # dataset
    full_dataset=Train_dataset(**kargs)
    
    num_samples=int(args.percent_samples*len(full_dataset))
    full_dataset=torch.utils.data.Subset(full_dataset,list(np.arange(num_samples)))

    #train test splite
    train_size = int(args.train_test_split * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    print(f'number train = {len(train_dataset):,}, number test = {len(test_dataset):,},batch size= {args.batch_size:,}')
    return train_dataset,test_dataset

def rename_conflict_filename(file_path):
    """
    Saves the given data to the given file path. If the file path already exists, renames the existing file
    to "old_<file_path>", and then saves the new data to the original file path.
    
    Arguments:
    file_path -- the path to the file to save the data to
    """
    if os.path.isfile(file_path):
        new_file_path = f"old_{file_path}"
        i = 1
        while os.path.isfile(new_file_path):
            new_file_path = f"old_{i}_{file_path}"
            i += 1
        print(f"rename {file_path} to {new_file_path}")
        os.rename(file_path, new_file_path)

def refine_best_model(target_model_name,target_rank_criteria='test MSE'):
    from ML_dmft.pytorch_model.loss_function import Loss_Function_Interface
    target_model_save_name = f'{target_model_name}_model.pth'

    for criteria in Loss_Function_Interface.methods():
        rank_criteria = f'test {criteria}'
        model_save_name = f'{target_model_name}_{criteria}_model.pth'

        path_list=glob.glob('./model_parameter/train/model_epoch_*.pth')
        metric_list=[]
        for path_checkpoint in path_list:
            checkpoint = torch.load(path_checkpoint)
            metric_list.append(checkpoint[rank_criteria])
        best_index=metric_list.index(min(metric_list))
        best_model=torch.load(path_list[best_index])
     
        rename_conflict_filename(model_save_name)
        print(50*'#')
        print(f'saving to {model_save_name=}')
        print(f'epoch   = {best_model["epoch"]}')
        print(f'{rank_criteria}={best_model[rank_criteria]}')
        print(f'test STD={best_model["test STD"]}')
        print(50*'#')
        torch.save(best_model,model_save_name)
        if rank_criteria == target_rank_criteria:
            rename_conflict_filename(target_model_save_name)
            print(f'saving to {target_model_save_name=}')
            torch.save(best_model,target_model_save_name)


def latest_iteration():
    print('Going to search partten ./model_parameter/train/model_epoch_*.pth')
    path_list=glob.glob('./model_parameter/train/model_epoch_*.pth')
    match_j=-1
    for path in path_list:
        file=path.split(os.sep)[-1]
        match=int(re.search('[0-9]+',file).group(0))
        if match > match_j:
            match_j=match
            latest_path=path
    print(f'\nthe latest iteration is {match_j} at {latest_path}')
    return latest_path

def save_checkpint(model,model_kargs,optimizer,scheduler,epoch,y_loss_dict):
    #TODO not only saving checkpoints also save CSV without such tools.
    time_start=perf_counter()
    
    checkpoint = {
        "epoch": epoch,
        'model_kargs':model_kargs,
        "net": model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'scheduler':scheduler.state_dict(),
    }
    for key in y_loss_dict: checkpoint[key]=y_loss_dict[key]

    if not os.path.isdir("./model_parameter/train"):
            os.makedirs("./model_parameter/train")
            print('mkdirs ./model_parameter/train')
    torch.save(checkpoint, './model_parameter/train/model_epoch_%s.pth' % (str(epoch)))
    print(f'\nsaved to {"./model_parameter/train/model_epoch_%s.pth" % (str(epoch))}')
    print(f'time to save the model {(perf_counter()-time_start):.5f}s.')
    return

def nullable_str(output_type,inputs:string):
    if inputs.lower() == 'none':
        return None
    else:
        return output_type(inputs)

class Nullable_Str:
    def __init__(self,o_type:Callable) -> None:
        self.o_type=o_type
    def __call__(self,inputs:str):
        try:
            if inputs.lower() in ['none','null']:
                return None
            else:
                return self.o_type(inputs)
        except:
            return self.o_type(inputs)

class EarlyStopper:
    def __init__(self, patience:int=1, min_delta:float=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
            

if __name__ == '__main__':
    import argparse
    from functools import partial
    parser = argparse.ArgumentParser(description='Anderson Transformer Example')
    parser.add_argument('--squasher_factor', type=partial(nullable_str,float), metavar='n',
                    help='squasher_factor to level up inputs',required=True)
    args = parser.parse_args()
    print(args.squasher_factor)
    print(type(args.squasher_factor))
