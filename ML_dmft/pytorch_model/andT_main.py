# -*- coding: utf-8 -*-
"""
Anderson Transformer main page
===============================

# Author: Zelong Zhao 2022
"""
import torch,os
from time import perf_counter
from .utility.args_parser_inputs import train_args
from .utility.train_tools import device_info,latest_iteration,save_checkpint,refine_best_model,checkpoints2csv,EarlyStopper
from .utility.optimizer_interface import optimizer_interface
from .utility.schedular_interface import Schedular_Interface
from .utility.loss_function import Loss_Function_Interface
from .utility.train_interface import train,train_val_evla
from .dataloader_tools import return_dataloader,dataloader_dim_interface
from .ML_models import model_interface,model_type2dataloader_input_type


def main():
    args=train_args()
    num_gpu=device_info()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #loading dataloader
    train_loader,test_loader,inverse_transform=return_dataloader(args)
    D_INPUT_TYPE=model_type2dataloader_input_type(args.model_type)
    dim_dict=dataloader_dim_interface(args.dataset_type,train_loader)
    model,model_kargs=model_interface(args.model_type,args,dim_dict)
    model=model(**model_kargs).to(device)

##############################################################################
    optimizer=optimizer_interface(args,model)
##############################################################################

##############################################################################
    SI=Schedular_Interface(args,optimizer)
    scheduler=SI.scheduler
##############################################################################

##############################################################################
#Define Loss functions 

    lf_interface = Loss_Function_Interface(args)
    criterion = lf_interface.criterion
    rank_criteria = f"test {args.criteria}"
##############################################################################
# initial model or reload
    start_epoch = 0

    if args.RESUME:
        # path_checkpoint = model_name # best one
        path_checkpoint=latest_iteration()
        checkpoint = torch.load(path_checkpoint)

        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for g in optimizer.param_groups:
            g['lr']=float(args.optimizer_args[1])

        start_epoch = checkpoint['epoch']+1
        # scheduler.load_state_dict(checkpoint['scheduler']) #NO need to load this again
        print(f'{start_epoch=}')
    
    else:

        if os.path.isdir("./model_parameter/train"):
            raise SyntaxError('./model_parameter/train exists, delete or specify RESUME')
        
        if args.RESUME_FILE is not None:
            print(f"Reloading from {args.RESUME_FILE=}")
            checkpoint = torch.load(args.RESUME_FILE)
            model.load_state_dict(checkpoint['net'])
            # optimizer.load_state_dict(checkpoint['optimizer'])

    last_time_finished=perf_counter()
    num_epochs=start_epoch+args.epochs

##############################################################################
    # EarlyStopper 
    es_dict=dict(patience=int(args.ES[0]),
                 min_delta=float(args.ES[1]),)
    long_text=f"""
es_dict:

{es_dict=}
"""
    print(long_text)
    early_stopper = EarlyStopper(**es_dict)
##############################################################################

    for epoch in range(start_epoch,start_epoch+args.epochs):
        #train
        time_last_epoch=perf_counter()
        if epoch == 0:
            print(f'{epoch} : running without train:')
            __train=False
        else:
            __train=True

        (actuals,predictions),(time_list_out,train_out_dict)=train(args=args,
                                                D_INPUT_TYPE=D_INPUT_TYPE,
                                                model=model,
                                                device=device,
                                                data_loader=train_loader,
                                                criterion=criterion,
                                                optimizer=optimizer,
                                                __train=__train,
                                                inverse_transform=inverse_transform,
                                                )

        if __train: scheduler.step()

        #epoch log interval
        if (epoch) % args.log_interval_epoch == 0:

            print('\n'+50*'#')
            
            print(f'epoch: {epoch}/{num_epochs-1}')
            print(f'tot time {(perf_counter()-time_last_epoch):.5f}s for {args.log_interval_epoch} epochs ')
            print(f'lr Scheduler={scheduler.get_last_lr()}')
            print('lr Optimiser=',optimizer.state_dict()['param_groups'][0]['lr'])
            print (f'\nnum worker {args.num_worker}') 
            for key in time_list_out:
                print(f"{key} {time_list_out[key]:.5f}s")
            print('\n')


            ##### test data
            time_counter_test=perf_counter()
            y_loss_dict=train_val_evla(model,device,train_out_dict,test_loader,args,D_INPUT_TYPE,inverse_transform)
            for key in y_loss_dict: print(f"{key} {y_loss_dict[key]}")
            print(f'time {(perf_counter()-time_counter_test):.5f}s for evaluating train and test')
            #####
            ## save
            save_checkpint(model,model_kargs,optimizer,scheduler,epoch,y_loss_dict)  

            if early_stopper.early_stop(y_loss_dict[rank_criteria]):             
                break
            print(50*'#'+'\n')

    print('\n'+50*'#')
    print(f'time to finish {(perf_counter()-last_time_finished):.5f} s')
    print(50*'#'+'\n')


    refine_best_model(target_model_name=args.model_name,
                    target_rank_criteria=rank_criteria)
    checkpoints2csv('./')



if __name__ == '__main__':
    main()