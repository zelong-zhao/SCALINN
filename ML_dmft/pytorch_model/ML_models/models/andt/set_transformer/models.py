# -*- coding: utf-8 -*-
"""
Set Transformer
===============

# Modifed From http://proceedings.mlr.press/v97/lee19d.html
"""

from .modules import *
from ..layers import MLP_PositionwiseFeedForward

class DeepSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128):
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, num_outputs*dim_output))

    def forward(self, X):
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X


class MeanPoolSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, depth,dropout=0.,dim_hidden=128):
        super(MeanPoolSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        
        self.enc = MLP_PositionwiseFeedForward(
                        in_feature=dim_input,
                        hidden_feature=dim_hidden,
                        out_feature=dim_hidden,
                        drop_prob=dropout,
                        depth=depth)
        self.dec = MLP_PositionwiseFeedForward(
                        in_feature=dim_hidden,
                        hidden_feature=dim_hidden,
                        out_feature=num_outputs*dim_output,
                        drop_prob=dropout,
                        depth=depth)

    def forward(self, X):
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))
