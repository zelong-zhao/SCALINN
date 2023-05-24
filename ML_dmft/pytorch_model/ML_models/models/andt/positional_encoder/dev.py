import torch
import torch.nn as nn 
from torch import nn, Tensor
from einops import rearrange, repeat

class Dev_PE(nn.Module):
    def __init__(self,beta:float,
                wn_s:int=None,wn_e:int=None,
                max_len:int=32,
                out_feature:int=512
                ) -> None:
        r"""
        inputs x
        x.size() = (b,seq_len,dim_val)
        """
        super().__init__()
        #assume 0~32
        self.wn_s = wn_s
        self.wn_e = wn_e
        matsu = (torch.pi * (1 + 2 * torch.arange(max_len)) / beta)

        # self.out_feature = out_feature-1
        # self.matsu_layer = Mlp_simple(in_feature=1,hidden_feature=self.out_feature ,out_feature=self.out_feature,dropout=0)

        # previous CODE
        self.register_buffer('matsu', matsu)
    
    def forward(self, x: Tensor,wn_s:int=None,wn_e:int=None) -> Tensor:
        b,t,_ = x.size()
        wn_s = wn_s or self.wn_s
        wn_e = wn_e or self.wn_e
        x = rearrange(x,'b t d -> 1 t (d b)')
        # pe = self.matsu[wn_s:wn_e].view(1,t,1)
        # pe = torch.flip(pe,[1])
        # pe = self.matsu_layer(pe)
        # pe = repeat(pe,'1 t d -> b t d',b=b)

        # x = torch.cat((x,pe),-1)
        return x


