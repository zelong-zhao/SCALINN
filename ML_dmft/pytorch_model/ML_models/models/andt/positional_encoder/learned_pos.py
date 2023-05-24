import torch
import warnings,os
from torch import nn, Tensor
from ML_dmft.utility.tools import get_bool_env_var
from einops import rearrange, repeat

class Learned_PositionalEncoder(nn.Module):
    def __init__(self,wn_s:int,wn_e:int,
                        max_len:int
                        ) -> None:
        r"""
        inputs x
        x.size() = (b,seq_len,dim_val)
 
        """
        super().__init__()

        self.DEBUG=get_bool_env_var('ML_dmft_DEBUG')

        self.wn_s = wn_s
        self.wn_e = wn_e
        self.pe_layer=nn.Parameter(torch.randn(max_len))

    def forward(self, x: Tensor,wn_s:int=None,wn_e:int=None) -> Tensor:
        if self.DEBUG: print(f"In PE add layer: {self.pe_layer=}")
        b,t,d = x.size()
        wn_s = wn_s or self.wn_s
        wn_e = wn_e or self.wn_e
        pe_layer = self.pe_layer[wn_s:wn_e].view(1,wn_e-wn_s,1)
        # pe = torch.cat(b*[pe_layer])
        x = x[:] + pe_layer
        return x
    

class Learned_PositionalEncoder_Concatenate(nn.Module):
    def __init__(self,wn_s:int,wn_e:int,
                        max_len:int,
                        dim_pos:int
                        ) -> None:
        r"""
        inputs x
        x.size() = (b,seq_len,dim_val)
 
        """
        super().__init__()

        self.DEBUG=get_bool_env_var('ML_dmft_DEBUG')

        self.wn_s = wn_s
        self.wn_e = wn_e
        self.dim_pos = dim_pos
        self.pe_layer=nn.Parameter(torch.randn(1,max_len,self.dim_pos))
        
        # self.pe_layer = nn.Parameter(torch.Tensor(1, max_len, self.dim_pos))
        # nn.init.xavier_uniform_(self.pe_layer)

    
    def forward(self, x: Tensor,wn_s:int=None,wn_e:int=None) -> Tensor:
        if self.DEBUG: print(f"In PE layer concatenate: {self.pe_layer=}")
        b,t,d = x.size()
        wn_s = wn_s or self.wn_s
        wn_e = wn_e or self.wn_e

        pe = self.pe_layer[:,wn_s:wn_e,:].view(1,t,self.dim_pos)
        pe = repeat(pe,'1 t d -> b t d',b=b)

        x = torch.cat((x,pe),-1)
        return x