import torch

def generate_square_subsequent_mask(dim1: int, dim2: int) -> torch.Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Source:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Args:
        dim1: int, for both src and tgt masking, this must be target sequence
              length
        dim2: int, for src masking this must be encoder sequence length (i.e. 
              the length of the input sequence to the model), 
              and for tgt masking, this must be target sequence length 
    Return:
        A Tensor of shape [dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

def rf_tgt_src_mask(dim1: int, input_seq_len=12,tot_seq_len=16,dec_seq_len=4) -> torch.Tensor:
    """
    tgt ~dec_seq_len
    src: tot_seq_len-input_seq_len:tot_seq_len
    """
    assert input_seq_len+dec_seq_len >= tot_seq_len
    assert dim1 <= dec_seq_len

    diag = min(tot_seq_len-dec_seq_len,input_seq_len)
    return torch.triu(torch.ones(dim1, input_seq_len) * float('-inf'), diagonal=diag)

def lookup_rf_tgt_src_mask(dim1: int, lookup_len=100,input_seq_len=12,tot_seq_len=16,dec_seq_len=8) -> torch.Tensor:
    """
    tgt ~dec_seq_len
    src: tot_seq_len-input_seq_len:tot_seq_len
    """
    assert input_seq_len+dec_seq_len >= tot_seq_len
    assert dim1 <= dec_seq_len

    diag = min(tot_seq_len-dec_seq_len,input_seq_len)
    mask =  torch.triu(torch.ones(dim1, input_seq_len), diagonal=diag)

    for idx,item in enumerate(mask):
        # item[max(diag-lookup_len+idx,0):dag+idx]=0
        item[0:max(diag-lookup_len+idx,0)]=1
        mask[idx]=item
    mask=(mask*float('-inf'))
    mask[mask != mask] = 0
    return mask


def gen_make_bidirectional_mask_rf_tgt_src(dim1: int,lookforward_len=1,lookbackward_len=100,input_seq_len=16,tot_seq_len=16,dec_seq_len=15) -> torch.Tensor:
    """
    tgt ~dec_seq_len
    src: tot_seq_len-input_seq_len:tot_seq_len
    """
    assert input_seq_len+dec_seq_len >= tot_seq_len
    assert dim1 <= dec_seq_len
    
    assert lookforward_len >= 0
    assert lookbackward_len >=0

    diag = min(tot_seq_len-dec_seq_len,input_seq_len)

    mask =  torch.triu(torch.ones(dim1, input_seq_len), diagonal=diag)
    for idx,item in enumerate(mask):
        item[0:min(diag+lookforward_len+idx,input_seq_len)]=0
        item[0:max(diag-lookbackward_len+idx,0)]=1
        mask[idx]=item
    mask=(mask*float('-inf'))
    mask[mask != mask] = 0
    return mask


def lookbackward_square_mask(seq_len:int,lookbackward_len:int)->torch.Tensor:
    mask=torch.ones(seq_len, seq_len)
    for idx,item in enumerate(mask):
        item[max(idx-lookbackward_len,0):idx+1]=0
        mask[idx]=item
    mask=(mask*float('-inf'))
    mask[mask != mask] = 0
    return mask
