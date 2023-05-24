from torch import nn
import torch

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            )
    def forward(self, x):
        return self.net(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden,act_layer=nn.GELU,drop_prob=0.):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.act1 = act_layer()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.dropout(x)
        return x

class Mlp(nn.Module):
    def __init__(self,in_feature,hidden_feature,out_feature=None,dropout=0,depth=3):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(FeedForward(in_feature,hidden_feature,dropout))
        for _ in range(depth):
            self.layers.append(FeedForward(hidden_feature,hidden_feature,dropout))
        self.out_layer=nn.Linear(hidden_feature,out_feature)

    def forward(self, x):
        for hiden_layer in self.layers:
            x=hiden_layer(x)

        x=self.out_layer(x)
        return x

class MLP_PositionwiseFeedForward(nn.Module):
    def __init__(self,in_feature, hidden_feature, out_feature,
                act_layer=nn.GELU,drop_prob=0.,depth=0):
        super().__init__()
        self.layers = nn.ModuleList([])

        if depth == 0:
            self.out_layer = None
            self.layers.append(PositionwiseFeedForward(in_feature,out_feature,act_layer,drop_prob))
        
        else:
            self.layers.append(PositionwiseFeedForward(in_feature,hidden_feature,act_layer,drop_prob))

            self.out_layer=PositionwiseFeedForward(hidden_feature,out_feature,
                                               act_layer=act_layer,drop_prob=drop_prob)  

        for _ in range(max(depth-1,0)):
            self.layers.append(PositionwiseFeedForward(hidden_feature,hidden_feature,act_layer,drop_prob))
        
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        for layer in self.layers:
            x=layer(x)

        if self.out_layer is not None:
            x=self.out_layer(x)
        return x

class Input_Encoder(nn.Module):
    def __init__(self,in_feature,hidden_feature=None,out_feature=None,act_layer=nn.GELU,dropout=0,depth=0) -> None:
        super().__init__()

        out_feature = out_feature or in_feature
        hidden_feature = hidden_feature or in_feature

        self.layer=MLP_PositionwiseFeedForward(in_feature,hidden_feature,
                                               act_layer=act_layer,drop_prob=dropout,
                                               depth=depth)
        
        self.out_layer=PositionwiseFeedForward(hidden_feature,out_feature,
                                               act_layer=act_layer,drop_prob=dropout)  
         
    def forward(self,x):
        x = self.layer(x)
        x = self.out_layer(x)
        return x


class Time_Step_Mlp(nn.Module):
    def __init__(self,in_feature:int,
                    out_feature:int,
                    time_step:int,
                    dim_hidden:int,
                    dropout:float=0.,
                    depth:int=0,
                    act_layer=nn.GELU) -> None:
        super().__init__()

        self.time_step = time_step
        self.out_feature = out_feature
        self.dim_hidden = dim_hidden
    
        stacked_output_features = self.dim_hidden*time_step

        self.fc1 = PositionwiseFeedForward(in_feature,stacked_output_features,
                                           act_layer=act_layer,drop_prob=dropout)

        self.layers = MLP_PositionwiseFeedForward(self.dim_hidden,self.dim_hidden,self.out_feature,
                                                drop_prob=dropout,depth=depth,
                                                act_layer=act_layer)
        
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        """
        input data shape
        x.size() = b,f

        output data shape

        x.size() = 
        """
        assert len(x.size())==2 
        b,_ = x.size()

        x = self.fc1(x)
        x = x.view(b,self.time_step,self.dim_hidden)

        x = self.layers(x)
            
        return x



class Mlp_simple(nn.Module):
    def __init__(self,in_feature,
                        hidden_feature=None,
                        out_feature=None,
                        act_layer=nn.GELU,
                        dropout=0.) -> None:
        super().__init__()
        out_feature = out_feature or in_feature
        hidden_feature = hidden_feature or in_feature
        self.fc1 = nn.Linear(in_feature,hidden_feature)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_feature,out_feature)
        self.drop = nn.Dropout(dropout)
    
    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
