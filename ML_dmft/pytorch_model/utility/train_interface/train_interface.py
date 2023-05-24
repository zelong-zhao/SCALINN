import torch
from time import perf_counter
import numpy as np
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from argparse import Namespace

from ML_dmft.utility.tools import get_bool_env_var
from .tool import list_dict_output,Mini_Batch_Pytorch
from .stas_method import Welford_Online_Stas


def train(args:Namespace,
        D_INPUT_TYPE:int,
        model,
        device:str,
        data_loader:DataLoader,
        criterion=None,
        optimizer:Optimizer=None,
        __train:bool=False,
        inverse_transform=None
        ):

    DEBUG = get_bool_env_var('ML_dmft_loss_DEBUG')

    if __train:
        model.train()
    else:
        model.eval()

    n_total_steps=len(data_loader)
    output_result = False
    if output_result:
        predictions, actuals = list(), list()
    else:
        predictions, actuals = None, None

    time_list=list()
    time_last=perf_counter()

    # initial time_counter
    data_loader_time_c=0
    time_forward_prop_c=0
    time_cal_loss_c=0

    # count loss
    max_loss,max_loss_init = -1*np.inf,-1*np.inf

    welford_stas = Welford_Online_Stas(args.Loss_args[0],inverse_transform)

    for batch_idx, ((inputs,target),_) in enumerate(data_loader): 

        if D_INPUT_TYPE==0:
            in_params = inputs
            in_params = in_params.to(device) 

        elif D_INPUT_TYPE in [1,2]:
            in_params,feature_G = inputs
            in_params = in_params.to(device) 
            feature_G = feature_G.to(device)

        elif D_INPUT_TYPE in [3]:
            param,seq=inputs
            glob_params,imp_params,hyb_params = param
            src,rf_tgt = seq

            glob_params,imp_params,hyb_params = glob_params.to(device) ,imp_params.to(device) ,hyb_params.to(device) 
            src,rf_tgt = src.to(device),rf_tgt.to(device)

        elif D_INPUT_TYPE in [4]:
            param,seq=inputs
            glob_params,imp_params,hyb_params,src_hyb_params = param
            src,tail_tgt,rf_tgt = seq

            glob_params,imp_params = glob_params.to(device) ,imp_params.to(device)
            hyb_params,src_hyb_params =  hyb_params.to(device),src_hyb_params.to(device)

            src,tail_tgt,rf_tgt = src.to(device),tail_tgt.to(device),rf_tgt.to(device)

        else:
            raise ValueError('D_INPUT_TYPE not found')

        target = target.to(device)
        batch_size = len(target)

        outputs = torch.zeros(target.size(),requires_grad=False).to(device)

        mini_batch_size = args.mini_batch_size 
        BM_indices= Mini_Batch_Pytorch(batch_size,mini_batch_size)

        data_loader_time=perf_counter()-time_last
        time_last=perf_counter()
        data_loader_time_c+=data_loader_time

        # Forward pass
        if __train:
            for inner_batch in range(len(BM_indices)):
                train_loop = args.train_loop

                for inner_loop in range(train_loop):
                    time_last=perf_counter()
                    
                    if D_INPUT_TYPE==0: 
                        outputs_one = model(in_params[BM_indices(inner_batch)])

                    elif D_INPUT_TYPE in [1,2]:

                        outputs_one = model(in_params[BM_indices(inner_batch)],
                                            feature_G[BM_indices(inner_batch)])
                        
                    elif D_INPUT_TYPE in [3]:

                        outputs_one = model(glob_params[BM_indices(inner_batch)],
                                            imp_params[BM_indices(inner_batch)],
                                            hyb_params[BM_indices(inner_batch)],
                                            src[BM_indices(inner_batch)],
                                            rf_tgt[BM_indices(inner_batch)])
                        
                    elif D_INPUT_TYPE in [4]:

                        outputs_one = model(glob_params[BM_indices(inner_batch)],
                                            imp_params[BM_indices(inner_batch)],
                                            hyb_params[BM_indices(inner_batch)],
                                            src_hyb_params[BM_indices(inner_batch)],
                                            src[BM_indices(inner_batch)],
                                            tail_tgt[BM_indices(inner_batch)],
                                            rf_tgt[BM_indices(inner_batch)])

                    else:
                        raise ValueError('D_INPUT_TYPE not found')
                    
                    time_forward_prop=perf_counter()-time_last
                    time_last=perf_counter()
                    time_forward_prop_c+=time_forward_prop
                    
                    target_1batch = target[BM_indices(inner_batch)]
                    
                    loss = criterion(outputs_one, target_1batch)  

                    if DEBUG: print(f"{loss=}")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
                    time_cal_loss=perf_counter()-time_last
                    time_last=perf_counter()
                    time_cal_loss_c+=time_cal_loss

                    if DEBUG:
                        for name,params in model.named_parameters():
                            print(f'-->name: {name} --> grad_requires -->{params.requires_grad} gard_dim--> {params.shape} \
                            \ngard_value-->max={torch.max(params.grad)} min={torch.min(params.grad)} \n ')

                    if args.dry_run:
                        print(f"Step:[{batch_idx+1}/{n_total_steps}] Mini-batch:{inner_batch+1}/{len(BM_indices)} {inner_loop}/{train_loop} lr:{optimizer.state_dict()['param_groups'][0]['lr']} Loss: {loss.item():.8f}'")
                    
                    max_loss_init=loss.item() if loss.item() > max_loss_init and inner_loop==0 else max_loss_init

                max_loss=loss.item() if loss.item() > max_loss else max_loss
                outputs[BM_indices(inner_batch)]=outputs_one

        else:
            with torch.no_grad():
                if D_INPUT_TYPE in [0,2]: 
                    outputs = model(in_params)
                elif D_INPUT_TYPE==1:
                    outputs = model(in_params,feature_G)
                elif D_INPUT_TYPE in [3]:
                    outputs = model(glob_params,imp_params,hyb_params,src,None)
                elif D_INPUT_TYPE in [4]:
                    outputs = model(glob_params,imp_params,hyb_params,src_hyb_params,
                                    src,tail_tgt,None)
                else:
                    raise ValueError('D_INPUT_TYPE not found')
            time_forward_prop=perf_counter()-time_last
            time_last=perf_counter()
            time_forward_prop_c+=time_forward_prop
        
        # store
        act,pred=target.cpu().data.numpy(),outputs.cpu().data.numpy()
        welford_stas(act,pred)

        if output_result:
            predictions.append(outputs)
            actuals.append(target)

        time_list={'data_loader_time':data_loader_time_c,
                    'time_forward_prop':time_forward_prop_c,
                    'time_cal_loss':time_cal_loss_c,
                    }  

        if (batch_idx+1) % args.log_interval == 0 or (batch_idx+1)==n_total_steps:
            if __train:
                print(50*'#')
                print(f"Step:[{batch_idx+1}/{n_total_steps}] Mini-batch:{inner_batch+1}/{len(BM_indices)} {inner_loop}/{train_loop} lr:{optimizer.state_dict()['param_groups'][0]['lr']} Loss: {loss.item():.8f}'")
                print(f"{max_loss=} {max_loss_init=}")
                print(f"{welford_stas.MAE=}")
                print(f"{welford_stas.MSE=}")
                print(f"{welford_stas.MATSU=}")
                print(f"{welford_stas.std=}")
                print (f'num worker {args.num_worker}') 
                print (f'{data_loader_time_c=:.5f}s')
                print (f'{time_forward_prop_c=:.5f}s')
                print (f'{time_cal_loss_c=:.5f}s')
                print(50*'#','\n')

            if args.dry_run:
                print('dry run, so exit')
                break
        
        time_last=perf_counter()


    # transfering data to CPU
    # diction raw_dict from torch to numpy
    if output_result:
        predictions=torch.cat(predictions).cpu().data.numpy()
        actuals=torch.cat(actuals).cpu().data.numpy()

    train_out_dict = {'MAE':welford_stas.MAE,
                    'MSE':welford_stas.MSE,
                    'Matsu':welford_stas.MATSU,
                    'STD':welford_stas.std}

    return (actuals,predictions),(time_list,train_out_dict)


def high_ram_train(args,D_INPUT_TYPE,model,device,data_loader,criterion=None,optimizer=None,__train=False,inverse_tranform=None):
    assert __train == False, 'meaningless for high_ram_train'

    model.eval()
    n_total_steps=len(data_loader)
    predictions, actuals = list(), list()
    raw_list=[]
    time_last=perf_counter()

    # initial time_counter
    data_loader_time_c=0
    time_forward_prop_c=0
    time_cal_loss_c=0

    for batch_idx, ((inputs,target),raw_dict) in enumerate(data_loader): 

        if D_INPUT_TYPE==0:
            in_params=inputs
            in_params = in_params.to(device) 
            
        elif D_INPUT_TYPE in [1,2]:
            in_params,feature_G = inputs
            in_params = in_params.to(device) 
            feature_G = feature_G.to(device)

        elif D_INPUT_TYPE in [3]:
            param,seq=inputs
            glob_params,imp_params,hyb_params = param
            src,rf_tgt = seq

            glob_params,imp_params,hyb_params = glob_params.to(device) ,imp_params.to(device) ,hyb_params.to(device) 
            src,rf_tgt = src.to(device),rf_tgt.to(device)

        elif D_INPUT_TYPE in [4]:
            param,seq=inputs
            glob_params,imp_params,hyb_params,src_hyb_params = param
            src,tail_tgt,rf_tgt = seq

            glob_params,imp_params = glob_params.to(device) ,imp_params.to(device)
            hyb_params,src_hyb_params =  hyb_params.to(device),src_hyb_params.to(device)

            src,tail_tgt,rf_tgt = src.to(device),tail_tgt.to(device),rf_tgt.to(device)


        else:
            raise ValueError('D_INPUT_TYPE not found')
        

        target = target.to(device)
        raw_list.append(raw_dict)

        data_loader_time=perf_counter()-time_last
        time_last=perf_counter()

        # Forward pass
        # if __train:
        #     if D_INPUT_TYPE==0: 
        #         outputs = model(in_params)
        #     elif D_INPUT_TYPE in [1,2]:
        #         outputs = model(in_params,feature_G)
        #     elif D_INPUT_TYPE in [3]:
        #         outputs = model(glob_params,imp_params,hyb_params,src,rf_tgt)
        #     else:
        #         raise ValueError('D_INPUT_TYPE not found')

        # else:
        with torch.no_grad():
            if D_INPUT_TYPE in [0,2]: 
                outputs = model(in_params)
            elif D_INPUT_TYPE == 1:
                outputs = model(in_params,feature_G)
            elif D_INPUT_TYPE in [3]:
                outputs = model(glob_params,imp_params,hyb_params,src,None)
            elif D_INPUT_TYPE in [4]:
                outputs = model(glob_params,imp_params,hyb_params,src_hyb_params,
                                    src,tail_tgt,None)
            else:
                raise ValueError('D_INPUT_TYPE not found')

        time_forward_prop=perf_counter()-time_last
        time_last=perf_counter()
        
        # Backward and optimize
        if __train:
            loss = criterion(outputs, target)  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # store
        predictions.append(outputs)
        actuals.append(target)
        
        time_cal_loss=perf_counter()-time_last

        data_loader_time_c+=data_loader_time
        time_forward_prop_c+=time_forward_prop
        time_cal_loss_c+=time_cal_loss

        time_list={'data_loader_time':data_loader_time_c,
                    'time_forward_prop':time_forward_prop_c,
                    'time_cal_loss':time_cal_loss_c}  

        if (batch_idx+1) % args.log_interval == 0:
            if __train:
                print(50*'#')
                print (f'Step [{batch_idx+1}/{n_total_steps}], Loss: {loss.item():.4f}')
                print (f'num worker {args.num_worker}') 
                print (f'{data_loader_time=:.5f}s')
                print (f'{time_forward_prop=:.5f}s')
                print (f'{time_cal_loss=:.5f}s')
                print(50*'#','\n')

            if args.dry_run:
                print('dry run, so exit')
                break
        
        time_last=perf_counter()


    # transfering data to CPU
        # diction raw_dict from torch to numpy
    for dict in raw_list:
        for key in dict:
            dict[key]=dict[key].data.numpy()

    predictions=torch.cat(predictions).cpu().data.numpy()
    actuals=torch.cat(actuals).cpu().data.numpy()

    # input row list
    time_step=len(predictions[0])
    batch_size=len(predictions)
    
    raw_list=list_dict_output(raw_list)
    
    for index,dict in enumerate(raw_list):
        if not inverse_tranform:
            dict['delta G prediction']=predictions[index].reshape(time_step,)
            dict['delta G target']=actuals[index].reshape(time_step,)
        else:
            dict['delta G prediction']=inverse_tranform(predictions[index].reshape(time_step,))
            dict['delta G target']=inverse_tranform(actuals[index].reshape(time_step,))
            for key in ['src','trg','trg_y','sequence']:
                #TODO: write here for doing Z score.
                dict[key]=inverse_tranform(dict[key])
    return raw_list


def train_val_evla(model,device,train_out_dict,test_loader,args,D_INPUT_TYPE,inverse_transform):
        
        (_,_),(_,test_out_dict)=train(args=args,
                                                D_INPUT_TYPE=D_INPUT_TYPE,
                                                model=model,
                                                device=device,
                                                data_loader=test_loader,
                                                criterion=None,
                                                optimizer=None,
                                                __train=False)
        y_loss_dict = {}
        for item in train_out_dict:
            y_loss_dict[f"train {item}"]=train_out_dict[item]

        for item in test_out_dict:
            y_loss_dict[f"test {item}"]=test_out_dict[item]

        return y_loss_dict