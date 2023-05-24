from ..loss_function.interface import Loss_Function_Interface
import torch
import glob
import os

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