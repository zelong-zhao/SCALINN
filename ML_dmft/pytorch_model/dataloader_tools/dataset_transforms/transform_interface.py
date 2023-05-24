from .method0 import method0_stacked_transform_xy,method0_Y_Inverse_Transform
from .method2 import method2_1_Param_G_Zscore_Inverse_Transform,method2_2_param_g_zscore_transform_two_solver
from ML_dmft.utility.tools import read_yaml
import inspect

class Transform_Interface:
    def __init__(self,transform_method:str,transform_args:list):
        Transform_Interface.help_info(transform_method,transform_args)

        if transform_method=='0':
            kargs=dict(squasher_factor=transform_args[0],
                    normalise_factor=transform_args[1],
                    squasher_factor_y=transform_args[2],
                    normalise_factor_y=transform_args[3])

            for key in kargs: print(f"{key} {kargs[key]}")
                
            self.transform=method0_stacked_transform_xy(**kargs)

            self.inverse_transform=method0_Y_Inverse_Transform \
                                (squasher_factor_y=kargs['squasher_factor_y'],
                                normalise_factor_y=kargs['normalise_factor_y'])

        elif transform_method=='2.2':

            if type(transform_args) == list:
                assert len(transform_args) ==1 ,'expect a filename only'
                file_name = str(transform_args[0])
            else:
                file_name = str(transform_args)
            
            transform_args = read_yaml(file_name)['transform-factors']
            for item in transform_args: print(f"{item:10s} {transform_args[item]}")
            print('\n')

            self.transform = method2_2_param_g_zscore_transform_two_solver(**transform_args)
            inverse_signature =list(inspect.signature(method2_1_Param_G_Zscore_Inverse_Transform).parameters)
            inverse_args = {item:transform_args[item] for item in inverse_signature}

            self.inverse_transform = method2_1_Param_G_Zscore_Inverse_Transform(**inverse_args)

        else:
            raise ValueError('norm method not found')
    
        print(50*"#")
    
    @staticmethod
    def help_info(transform_method:str=None,transform_args:list=None):
        long_text=f"""
{50*'#'}
dataset transforms

Options
-------
0:   Squasher X,Y, log(X,Y) then Norm(X,Y). 
     X and Y are found via full time-step Error correction dataset.
    

2.2  Sample => (src,rf_tgt_tail,rf_tgt),tgt and glob_param,imp_param,hyb_param,src_hyb_param
transform method   
                0 = None 
                1 = normalise
                2 = standardise 
                3 = merged_noramlise (aim params) / *-1 normalise (G_iw)
                4 = merged_standardise (aim params) / *-1 standardise (G_iw)

self.transform =  enc_seq_len,trg_seq_len,Z_score(X)
self.inverse_transform(X) = flip(Z_score(X))

{transform_method=} is Choosen
{transform_args}
{50*'#'}
"""
        print(long_text)
