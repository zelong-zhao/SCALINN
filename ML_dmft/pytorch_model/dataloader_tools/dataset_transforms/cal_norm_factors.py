from . import method0_cal_norm_factor,method1_cal_norm_factor
from .tools import write_to_csv

def cal_norm_factor(norm_method:str,dataset_kargs:dict)->None:
    """
    calculate normalisation factors
    """

    long_text=f"""
{50*'#'}
Calculate Norm factors.

Options
-------
0:  Squasher X,Y, log(X,Y) then Norm(X,Y). 
    X and Y are found via full time-step Error correction dataset.
    
1:  Squasher X,Y then Norm(X,Y). 
    X and Y are found via full time-step Error correction dataset


{norm_method=} is Choosen
{dataset_kargs}
{50*'#'}
"""
    print(long_text)
    out_dict={}
    out_dict['dataset_method']=dataset_kargs['method']
    out_dict['norm_method']=norm_method


    if norm_method=="0":
        _out_dict=method0_cal_norm_factor(**dataset_kargs)

    elif norm_method=='1':
        _out_dict=method1_cal_norm_factor(**dataset_kargs)

    else:
        raise ValueError('norm_method is not recongnised')

    for key in _out_dict: out_dict[key]=_out_dict[key]
    write_to_csv(out_dict)