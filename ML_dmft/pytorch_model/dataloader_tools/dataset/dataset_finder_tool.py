from .Encoded_Param_Dataset import Encoded_Param_Dataset

def dataset_interface_help_info(type_dataset:str=None):
    long_txt=f"""
{50*'#'}
dataset interface:

Option
------

0: d3mft (developing)
6: encoded_param_transformer

-------
item format

0: D3MFT
G,delta_G = Sample 

6: Encoded_Param_Sigma_Dataset
(src,rf_tgt),tgt = Sample
(glob_param,imp_param,hyb_param,src,rf_tgt),tgt = Sample
--------

{type_dataset} is choose as dataset
{50*'#'}
    """
    print(long_txt)

def dataset_interface(type_dataset:str,args,transform):
    
    dataset_interface_help_info(type_dataset)

  
    if type_dataset=='6':

        dataset_kargs=dict(db=args.db,
        root=args.root_database,
        solver=args.solver,
        basis=args.basis,
        solver_j=args.solver_j,
        transform_dict=transform,
        )

        assert type(transform) == dict, 'transform type should be dictionary'

        return Encoded_Param_Dataset(**dataset_kargs)


    else:

        ValueError('type dataset not found')


def dataloader_dim_interface(type_dataset:str,dataloader)->dict:
    print(50*"#",'\ndataloader dim_interface:\n')
    
    if type_dataset=="2":
        loader=iter(dataloader)
        (param,sample_G),_=next(loader)
        dim=param.shape[1]
        time_steps=sample_G.shape[1]
        print(f"input dim={dim} time-steps={time_steps}")
        out_dict=dict(dim=dim,
                    time_steps=time_steps)

    elif type_dataset=="3":
        loader=iter(dataloader)
        inputs,_=next(loader)
        (x,xx),target=inputs
        x_ts,x_feature=x.shape[1:]
        xx_ts,xx_feature=xx.shape[1:]
        target_ts,target_feature=target.shape[1:]
        
        out_dict=dict(x_ts=x_ts,x_feature=x_feature,
                    xx_ts=xx_ts,xx_feature=xx_feature,
                    target_dim=target_ts,target_feature=target_feature)

        print(out_dict)

    elif type_dataset=="4":
        loader=iter(dataloader)
        inputs,_=next(loader)
        (x,xx),target=inputs
        print(x.shape,xx.shape,target.shape)
        x_ts,x_feature=x.shape[1:]
        xx_ts,xx_feature=xx.shape[1:]
        target_ts,target_feature=target.shape[1:]
        
        out_dict=dict(x_ts=x_ts,x_feature=x_feature,
                    xx_ts=xx_ts,xx_feature=xx_feature,
                    target_dim=target_ts,target_feature=target_feature)

        print(out_dict)

    elif type_dataset=="5":
        loader=iter(dataloader)
        inputs,data_dict=next(loader)
        (param,seq),target=inputs
        glob_params,imp_params,hyb_params = param
        beta=glob_params[0,1]
        print(f"{beta=}")
        print(f"{glob_params.shape=},{imp_params.shape=},{hyb_params.shape=}")


        glob_param_size = glob_params.shape[1]
        imp_param_size = imp_params.shape[2]
        hyb_param_ts,hyb_param_size = hyb_params.shape[1:]

        src,rf_tgt = seq
        print(f"{src.shape=} {rf_tgt.shape=} {target.shape}")
        src_ts,src_feature = src.shape[1:]
        rf_tgt_ts,rf_tgt_feature = rf_tgt.shape[1:]
        target_ts,target_feature=target.shape[1:]

        print(f"{data_dict['sequence'].shape=}")
        seq_ts=data_dict['sequence'].shape[1]
 
        out_dict=dict(beta=beta,
                    glob_param_size=glob_param_size,imp_param_size=imp_param_size,
                    hyb_param_ts=hyb_param_ts,hyb_param_size=hyb_param_size,
                    src_ts=src_ts,src_feature=src_feature,
                    rf_tgt_ts=rf_tgt_ts,rf_tgt_feature=rf_tgt_feature,
                    target_ts=target_ts,target_feature=target_feature,
                    seq_ts=seq_ts,
                    )
        print(out_dict)

    elif type_dataset=="6":
        loader=iter(dataloader)
        inputs,data_dict=next(loader)
        (param,seq),target=inputs
        glob_params,imp_params,hyb_params,src_hyb_params= param

        beta=glob_params[0,1]
        print(f"{beta=}")
        print(f"{glob_params.shape=}\n{imp_params.shape=}\n{hyb_params.shape=}\n{src_hyb_params.shape=}")

        glob_param_size = glob_params.shape[1]
        imp_param_size = imp_params.shape[2]
        num_tgt_solver,hyb_param_ts,hyb_param_size = hyb_params.shape[1:]
        assert num_tgt_solver == 1

        src,tail,rf_tgt = seq
        print(f"\n{src.shape=}\n{tail.shape=}\n{rf_tgt.shape=}\n{target.shape=}")
        src_ts,src_feature = src.shape[1:]
        rf_tgt_ts,rf_tgt_feature = rf_tgt.shape[1:]
        target_ts,target_feature=target.shape[1:]

        tail_ts = tail.shape[1]
        print(f"{tail_ts=}")

        print(f"{data_dict['sequence'].shape=}")
        seq_ts=data_dict['sequence'].shape[1]
 
        out_dict=dict(beta=beta,
                    glob_param_size=glob_param_size,
                    imp_param_size=imp_param_size,
                    hyb_param_size=hyb_param_size,
                    src_ts=src_ts,src_feature=src_feature,
                    rf_tgt_ts=rf_tgt_ts,rf_tgt_feature=rf_tgt_feature,
                    target_ts=target_ts,target_feature=target_feature,
                    seq_ts=seq_ts,
                    tail_ts=tail_ts
                    )

        print(out_dict)
    else: 
        raise ValueError('type_dataset not found')
    print(f"printing out_dict")
    for item in out_dict:
        print(f"{item}:{out_dict[item]}")
    print(50*'#')
    
    return out_dict