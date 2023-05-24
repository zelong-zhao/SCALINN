from .dataset_transforms import Transform_Interface
from torch.utils.data import DataLoader
from .dataset import dataset_interface,dataset_splite,load_part_dataset,Random_Seed_Interface

def return_dataloader(args):
    #dataset
    TI = Transform_Interface(transform_method=args.transform_method,
                                transform_args=args.transform_factors)
    transform=TI.transform
    inverse_transform=TI.inverse_transform

    
    loaded_dataset = dataset_interface(type_dataset=args.dataset_type,
                                        args=args,
                                        transform=transform)
    train_dataset,test_dataset = dataset_splite(args,loaded_dataset)

    #random seed:
    RSI=Random_Seed_Interface(args.seed)

    # dataloader
    pin_memory = False
    train_loader = DataLoader(dataset=train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.num_worker,
                                pin_memory=pin_memory,
                                worker_init_fn=RSI.seed_worker,
                                generator=RSI.Generator,
                                )

    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_worker,
                            pin_memory=pin_memory,
                            worker_init_fn=RSI.seed_worker,
                            generator=RSI.Generator,
                            )
                            
    print(f'Num step each epoch {len(train_loader):,}')
    return train_loader,test_loader,inverse_transform

def return_dataloader_and_inverse_transform(args):
    r"""
    for prediction
    """
    TI = Transform_Interface(transform_method=args.transform_method,
                                transform_args=args.transform_factors)
    transform=TI.transform
    inverse_transform=TI.inverse_transform

    loaded_dataset=dataset_interface(type_dataset=args.dataset_type,args=args,transform=transform)
    loaded_dataset=load_part_dataset(args.num_samples,loaded_dataset)

    # dataloader
    pin_memory=False

    test_loader = DataLoader(dataset=loaded_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_worker,
                            pin_memory=pin_memory)

    return test_loader,inverse_transform