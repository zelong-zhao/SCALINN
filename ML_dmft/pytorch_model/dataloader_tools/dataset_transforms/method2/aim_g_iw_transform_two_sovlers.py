import torch
import numpy as np
from ML_dmft.database.dataset import AIM_Dataset
from ML_dmft.pytorch_model.utility.train_tools.tool import Nullable_Str

class ToTensor:
    def __init__(self, precision=None):
        self.precision = precision
    # Convert ndarrays to Tensors
    def __call__(self, inputs):
        if self.precision:
            return torch.from_numpy(inputs.copy()).type(self.precision)
        else:
            return torch.from_numpy(inputs)


class Opposite_Zscore_Interface:
    def __init__(self,obj,forward:bool) -> None:
        self.obj=obj
        self.forward=forward
    def __call__(self, x ):
        if self.forward:
            x = self.obj(x*(-1))
        else:
            x = self.obj(x)
            x = x*(-1)
        return x
        

def param_g_zscore_transform(tail_len:int,
                            hyb_drop_solver_idx:int,
                            g_iw_method:int,
                            beta_method:int,mu_method:int,
                            U_method:int,eps_method:int,
                            E_k_method:int,V_k_method:int,
                            root_db:str,db:str,solver:str,basis:str)->dict:

    totensor=ToTensor(torch.float)
    dataset=AIM_Dataset(root_db,db,solver)
    
    transform_method_list=[None,'normalise','standardise','normalise','standardise']

    # G_Zscore
    method = transform_method_list[g_iw_method]
    key = basis
    if method is not None:
        print(f"{g_iw_method=}:{method} {key}")
        G_Zscore = dataset.data_transform(key,method,True)
        if g_iw_method in [3,4]:
            G_Zscore = Opposite_Zscore_Interface(G_Zscore,True)
    else:
        G_Zscore = None

    # beta_Zscore
    assert beta_method == 0, 'glob param transform havent developed yet'
    method = transform_method_list[beta_method]
    key='beta'
    print(f"{beta_method=}:{method} {key}")
    beta_Zscore = None

    # mu_Zscore
    assert mu_method == 0, 'glob param transform havent developed yet'
    method = transform_method_list[mu_method]
    key='mu'
    print(f"{mu_method=}:{method} {key}")
    mu_Zscore = None

    # U_Zscore'
    assert U_method == eps_method
    method = transform_method_list[U_method]
    if method is not None:
        if U_method < 2:
            key = "U"
        else:
            key = "U_eps_tot"
    
        print(f"{U_method=}:{method} {key}")
        U_Zscore = dataset.data_transform(key,method,True)

    else:
        U_Zscore = None

    # eps_Zscore

    method = transform_method_list[eps_method]
    if method is not None:
        if U_method < 2:
            key = "eps"
        else:
            key = "U_eps_tot"
    
        print(f"{eps_method=}:{method} {key}")
        eps_Zscore = dataset.data_transform(key,method,True)

    else:
        eps_Zscore = None

    # E_k_Zscore

    assert E_k_method == V_k_method
    
    method = transform_method_list[E_k_method]
    if method is not None:
        if U_method < 2:
            key = "E_k"
        else:
            key = "E_k_V_k_tot"
    
        print(f"{E_k_method=}:{method} {key}")
        E_k_Zscore = dataset.data_transform(key,method,True)

    else:
        E_k_Zscore = None

    # V_k_Zscore
    method = transform_method_list[V_k_method]
    if method is not None:
        if U_method < 2:
            key = "V_k"
        else:
            key = "E_k_V_k_tot"
    
        print(f"{V_k_method=}:{method} {key}")
        V_k_Zscore = dataset.data_transform(key,method,True)

    else:
        V_k_Zscore = None


    transform_dict = {'ToTensor':totensor,
                    'tail_len':tail_len,
                    'hyb_drop_solver_idx':Nullable_Str(int)(hyb_drop_solver_idx),
                    'G_Zscore':G_Zscore,
                    'beta_Zscore':beta_Zscore,'mu_Zscore':mu_Zscore,
                    'U_Zscore':U_Zscore,'eps_Zscore':eps_Zscore,
                    'E_k_Zscore':E_k_Zscore,'V_k_Zscore':V_k_Zscore
                    }   

    return transform_dict