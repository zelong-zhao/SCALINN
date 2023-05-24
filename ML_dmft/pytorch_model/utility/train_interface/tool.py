import numpy as np
import torch

def transfer2onebatch(x:torch.Tensor,batch_index:int):
    return torch.unsqueeze(x[batch_index,:],0)

class Mini_Batch_Pytorch:
    def __init__(self,batch_size:int,mini_batch_size:int) -> None:
        self.batch_size=batch_size
        self.mini_batch_size=mini_batch_size
        self.index_list = torch.arange(self.batch_size)

    def __call__(self, idx) :
        return self.index_list[idx*self.mini_batch_size:(idx+1)*self.mini_batch_size]
    
    def __len__(self) -> int:
        if self.batch_size % self.mini_batch_size == 0:
            return int(self.batch_size/self.mini_batch_size)
        else:
            return int(self.batch_size/self.mini_batch_size)+1

def list_dict_output(raw_list):
    keys=list(raw_list[0].keys())

    out_dict={}
    for key in keys:
        temp_list=[]
        for dict in raw_list:
            temp_list.append(dict[key])
        temp_list=np.vstack(temp_list)
        out_dict[key]=temp_list

    batch_size=len(out_dict[keys[0]])
    out_list=[]
    for index in range(batch_size):
        _dict={}
        for key in keys:
            if key == 'i' or key=='j' or key == 'index':
                _dict[key]=int(out_dict[key][index][0])
            elif key in ['G_i','G_j','src','trg','trg_y','sequence']:
                # print(f"in list dict output {key} {out_dict[key][index].shape=}")
                _dict[key]=out_dict[key][index].reshape(-1)
            else:
                
                _dict[key]=out_dict[key][index].reshape(-1)
        out_list.append(_dict)
    
    dict=out_list[0]
    print(50*'#')
    print('output data shape')
    for key,item in dict.items():
        try:
            print(f"{key=} {item.shape=}")
        except:
            print(f"{key=} {item=}")
    print(50*'#')
    return out_list