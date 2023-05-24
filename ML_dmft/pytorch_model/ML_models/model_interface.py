from .models import And_Two_Solver_Transformer
from ML_dmft.utility.tools import read_yaml
import contextlib

def model_interface_help_info(type_model:str=None):
    long_txt=f"""
{50*'#'}
loading model: 

Option
------

0: D3MFT (developing)
7: Transformer


{type_model} is choose as model
{50*'#'}
    """
    print(long_txt)

def model_interface(type_model,args=None,dim_dict:dict=None):
    r"""
Inputs
------
type_model

Return
------
Initialised model
"""

    model_interface_help_info(type_model)
    model_kargs=None

    if type_model == '7':
        if args and dim_dict:
            if type(args.model_hyper_params) == list:
                assert len(args.model_hyper_params) ==1 ,'expect a filename only'
            print(args.model_hyper_params)
            
            if args.model_hyper_params is not None:
                model_parameter = read_yaml(str(args.model_hyper_params[0]))['model-params']

                model_kargs=dict(beta=dim_dict['beta'],
                                input_seq_size=dim_dict['src_feature'],
                                input_seq_len=dim_dict['src_ts'],
                                dec_seq_len=dim_dict['rf_tgt_ts'],
                                tot_seq_len=dim_dict['seq_ts'],
                                glob_param_size=dim_dict['glob_param_size'],
                                imp_param_size=dim_dict['imp_param_size'],
                                hyb_param_size=dim_dict['hyb_param_size'],
                                tail_seq_len=dim_dict['tail_ts'],
                                )
                
                for item in model_parameter: model_kargs[item]=model_parameter[item]

                print(50*'#')
                for item in model_kargs:
                    print(f"{item}: {model_kargs[item]}")
                print(50*'#')

        model = And_Two_Solver_Transformer


    else:
        raise ValueError('type model not found!')
    
    if model_kargs is not None:
        with open('model_info.txt', 'a') as f,contextlib.redirect_stdout(f):
            print(50*'#')
            for item in model_kargs:
                print(f"{item}: {model_kargs[item]}")
            print(50*'#')


    return model,model_kargs

def model_type2dataloader_input_type(model_type):
    long_text=f"""
{50*'#'}
model_type2dataloader_input_type
input
-----
model_type
0
1
2
3 for stacked inputs
-----
output
-----
0 for inputs,targets=sample
1 for (inputs1,inputs2),target=sample

2:
training (inputs1,inputs2),target=sample
evaluate (inputs1,_),target=sample
"""
    print(long_text)
    
    if model_type in str([7]):
        D_INPUT_TYPE = 4
    else:
        raise ValueError('model_type not found')
    print(f"{D_INPUT_TYPE=} is identified")
    print(50*'#')  
    return D_INPUT_TYPE