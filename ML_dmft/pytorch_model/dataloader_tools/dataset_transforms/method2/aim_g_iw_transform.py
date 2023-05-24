import torch
import numpy as np
from ML_dmft.database.dataset import AIM_Dataset

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
 
class Param_G_Zscore_Inverse_Transform:
    def __init__(self,g_iw_method,root_db:str,db:str,solver:str,basis:str) -> None:
        transform_method_list=[None,'normalise','standardise','normalise','standardise']
        dataset=AIM_Dataset(root_db,db,solver)
        # G_Zscore
        method = transform_method_list[g_iw_method]
        key = basis

        if method is not None:
            print(f"Param_G_Zscore_Inverse_Transform: {g_iw_method=}:{method} {key}")
            self.G_Zscore = dataset.data_transform(key,method,False)
            if g_iw_method in [3,4]:
                self.G_Zscore = Opposite_Zscore_Interface(self.G_Zscore,False)
        else:
           self.G_Zscore = None

    def __call__(self,targets):
        targets=np.array(targets)
        targets=np.flip(targets)
        if self.G_Zscore is not None:
            targets=self.G_Zscore(targets)
        return  targets