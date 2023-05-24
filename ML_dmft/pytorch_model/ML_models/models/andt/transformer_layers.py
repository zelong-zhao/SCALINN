import torch
from torch import nn
import os
import math

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / (std + self.eps)
        out = self.gamma * out + self.beta
        return out

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden,act_layer=nn.GELU,drop_prob=0.):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.act1 = act_layer()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention
    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score+mask

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.DEBUG = bool(int(os.getenv('ML_dmft_DEBUG')))

        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        if self.DEBUG: print(f"In MultiHeadAttention before Linear: {q.shape=} {k.shape=} {v.shape=}")

        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        if self.DEBUG: print(f"In MultiHeadAttention after Linear: {q.shape=} {k.shape=} {v.shape=}")


        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)
        if self.DEBUG: print(f"In MultiHeadAttention after split heads: {q.shape=} {k.shape=} {v.shape=}")

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        if self.DEBUG: print(f"In MultiHeadAttention after calculate attention: {out.shape=}")

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        if self.DEBUG: print(f"In MultiHeadAttention after concat attention: {out.shape=}")
        # 5. visualize attention map
        # TODO : we should implement visualization
        return out

    def split(self, tensor):
        """
        split tensor by number of head
        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)
        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class EncoderLayer(nn.Module):

    def __init__(self, d_model:int, ffn_hidden:int, n_head:int, drop_prob:float,act_layer,norm=None):
        super(EncoderLayer, self).__init__()

        self.DEBUG = bool(int(os.getenv('ML_dmft_DEBUG')))

        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)

        self.norm1 = norm(d_model) if norm is not None else None

        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden,act_layer=act_layer, drop_prob=drop_prob)

        self.norm2 = norm(d_model) if norm is not None else None

        self.dropout2 = nn.Dropout(p=drop_prob)

        
    def forward(self, x, s_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=s_mask)
        
        # 2. add and norm

        x = self.norm1(x) if self.norm1 is not None else x + _x
        x = self.dropout1(x)
        
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
      
        # 4. add and norm
        x = self.norm2(x + _x) if self.norm2 is not None else x + _x
        x = self.dropout2(x)
        return x

class DecoderLayer(nn.Module):

    def __init__(self, d_model:int, ffn_hidden:int, n_head:int, drop_prob:float,act_layer,norm=None):
        super(DecoderLayer, self).__init__()
        self.DEBUG = bool(int(os.getenv('ML_dmft_DEBUG')))

        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = norm(d_model) if norm is not None else None
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = norm(d_model) if norm is not None else None
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden,act_layer=act_layer,drop_prob=drop_prob)
        self.norm3 = norm(d_model) if norm is not None else None
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, t_mask, s_mask):
        # 1. compute self attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=t_mask)
        
        # 2. add and norm
        x = self.norm1(x + _x) if self.norm1 is not None else x + _x
        x = self.dropout1(x)

        if enc is not None:
            # 3. compute encoder - decoder attention
            _x = x
            if self.DEBUG: print(f"In decoder layer, before enc_dec_attention: {x.shape=} {enc.shape=}")
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=s_mask)
            if self.DEBUG: print(f"In decoder layer, after enc_dec_attention: {x.shape=}")

            # 4. add and norm
            x = self.norm2(x + _x) if self.norm2 is not None else x + _x
            x = self.dropout2(x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.norm3(x + _x) if self.norm3 is not None else x + _x
        x = self.dropout3(x)
        return x


class TransforemerEncoder(nn.Module):
    def __init__(self,depth,d_model, ffn_hidden, n_head, drop_prob,act_layer,norm=None) -> None:
        super(TransforemerEncoder,self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(EncoderLayer(d_model=d_model,
                                            ffn_hidden=ffn_hidden,
                                            n_head=n_head,
                                            drop_prob=drop_prob,
                                            act_layer=act_layer,
                                            norm=norm)
                                            )
    
    def forward(self,x:torch.Tensor,mask:torch.Tensor=None)->torch.Tensor:
        for layer in self.layers:
            x = layer(x,mask)
        return x

class TransforemerDecoder(nn.Module):
    def __init__(self, d_model, num_predicted_features, ffn_hidden, n_head, n_decoder_layers, drop_prob,act_layer,norm=None):
        super().__init__()

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob,
                                                  norm=norm,
                                                  act_layer=act_layer
                                                  )
                                     for _ in range(n_decoder_layers)])

        self.linear = nn.Linear(d_model, num_predicted_features)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)
        return output