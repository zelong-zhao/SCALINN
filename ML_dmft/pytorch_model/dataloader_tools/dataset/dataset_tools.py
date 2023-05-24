import torch 
import numpy as np
import random

class Random_Seed_Interface:
    def __init__(self,seed) -> None:
        self.seed = seed
            
        if self.seed is None:
            self.Generator = None
        else:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

            self.Generator = torch.Generator()
            self.Generator.manual_seed(seed)

    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)



def dataset_splite(args,full_dataset):
    # dataset
    print(f'dataset splite trian and test :{args.percent_samples=} {args.train_test_split=}')
    if int(args.train_test_split) != 1:
        num_samples=int(args.percent_samples*len(full_dataset))
        full_dataset=torch.utils.data.Subset(full_dataset,list(np.arange(num_samples)))

        #train test splite
        train_size = int(args.train_test_split * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    else:
        train_dataset=full_dataset
        test_dataset=full_dataset

    print(f'number train = {len(train_dataset):,}, number test = {len(test_dataset):,},batch size= {args.batch_size:,}')
    return train_dataset,test_dataset

def load_part_dataset(num_samples,dataset):
    r"""
    dataset torch dataset type
    """
    if num_samples == -1:
        return torch.utils.data.Subset(dataset,list(np.arange(len(dataset))))
    else:
        return torch.utils.data.Subset(dataset,list(np.arange(min(num_samples,len(dataset)))))