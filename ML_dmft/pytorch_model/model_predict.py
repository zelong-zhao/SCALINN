import torch
from .dataloader_tools import return_dataloader_and_inverse_transform,dataloader_dim_interface
from .ML_models.model_interface import model_interface,model_type2dataloader_input_type
from .utility.train_interface import high_ram_train

def model_evaluate(args,model_path):
    r"""
    model evaluate: 

    load model_path and evaluate the module for specific tasks.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #loading dataloader
    test_loader,inverse_transform=return_dataloader_and_inverse_transform(args)
    D_INPUT_TYPE=model_type2dataloader_input_type(args.model_type)

    checkpoint = torch.load(model_path)
    print(f"As args.model_hyper_params is not None, loading new parameters to model.")
    dim_dict=dataloader_dim_interface(args.dataset_type,test_loader)
    model,model_kargs=model_interface(args.model_type,args,dim_dict)
    
    if args.model_hyper_params is not None:
        model=model(**model_kargs).to(device)
    # loading model
    else:
        model=model(**checkpoint['model_kargs']).to(device)
    


    model.load_state_dict(checkpoint['net'])
    
    Predicted_G_dict=high_ram_train(args=args,
                                        D_INPUT_TYPE=D_INPUT_TYPE,
                                        model=model,
                                        device=device,
                                        data_loader=test_loader,
                                        criterion=None,
                                        optimizer=None,
                                        __train=False,
                                        inverse_tranform=inverse_transform
                                        ) 
    return Predicted_G_dict