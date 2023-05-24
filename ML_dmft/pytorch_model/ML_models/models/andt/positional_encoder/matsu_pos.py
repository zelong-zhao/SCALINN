import torch
import torch.nn as nn 
from torch import nn, Tensor

class Matsubara_PositionalEncoder_Concatenate(nn.Module):
    def __init__(self,beta:float,
                wn_s:int=None,wn_e:int=None,
                max_len:int=32,
                transform_method:str='normalised'
                ) -> None:
        r"""
        inputs x
        x.size() = (b,seq_len,dim_val)
        """
        super().__init__()
        #assume 0~32
        self.wn_s = wn_s
        self.wn_e = wn_e

        transform_method_types=['None','standardised','normalised','normalised-standardised']
        if transform_method not in transform_method_types:
            raise ValueError('transform_method not found')

        matsu = (torch.pi * (1 + 2 * torch.arange(max_len)) / beta)
        matsu_mean = torch.mean(matsu)
        matsu_std = torch.std(matsu)
        standardlised_matsu = (matsu-matsu_mean)/matsu_std

        matsu_max = torch.max(matsu)
        matsu_min = torch.min(matsu)
        normalised_matsu = (matsu-matsu_min)/(matsu_max-matsu_min)

        matsu_normalised_mean = torch.mean(normalised_matsu)
        matsu_normalised_std = torch.std(normalised_matsu)
        standardlised_normalised_matsu = (matsu - matsu_normalised_mean)/matsu_normalised_std
        
        # print(f"assume from [0~31]")
        # print(f"{omega_mean=} {omega_std=}")
        if transform_method == 'standardised':
            pe_matsu = standardlised_matsu
        elif transform_method == 'normalised':
            pe_matsu = normalised_matsu
        elif transform_method == 'normalised-standardised':
            pe_matsu = standardlised_normalised_matsu
        elif transform_method == 'None':
            pe_matsu = matsu
        else:
            raise AssertionError('no such transfrom method')

        print(f"{transform_method=}\n{pe_matsu=}")
        omegan_list = pe_matsu

        # previous CODE
        self.register_buffer('omegan_list', omegan_list)
    
    def forward(self, x: Tensor,wn_s:int=None,wn_e:int=None) -> Tensor:
        b,t,d = x.size()
        wn_s = wn_s or self.wn_s
        wn_e = wn_e or self.wn_e

        omegan_layer = self.omegan_list[wn_s:wn_e].view(1,wn_e-wn_s,1)
        omegan_layer = torch.flip(omegan_layer,[1])

        # print(f"{self.omegan_layer[0].T=}")
        
        pe = torch.cat(b*[omegan_layer])

        x = torch.cat((x,pe),-1)
        return x


