import torch
from torch import Tensor
from torch import nn
from .layers import Time_Step_Mlp
from ML_dmft.utility.tools import get_bool_env_var
from .set_transformer import DeepSet,SetTransformer,MeanPoolSet
from einops import rearrange


class Error_Correction_Encoded_Param_Layer(nn.Module):
    def __init__(self,glob_param_size:int,
                    imp_param_size:int,
                    hyb_param_size:int,
                    dim_val:int,
                    dropout_mlp:float,
                    src_len:int,
                    rf_tgt_len:int,
                    cat2pe:bool,
                    cat2kv:bool,
                    cat2kv_hyb_pool:str='cat',
                    num_heads:int=4,
                    depth:int=4,
                    glob_param_outsize:int=32,
                    imp_param_outsize:int=32,
                    hyb_param_outsize:int=128,
                    src_hyb_param_outsize:int=128,
                    hyb_use_transformer:bool=False,
                    sep_glob_layer:bool=True,
                    sep_imp_layer:bool=True,
                    sep_hyb_layer:bool=True,
                    ) -> None:
        super().__init__()
        
        print(50*"#")
        self.DEBUG = get_bool_env_var('ML_dmft_DEBUG')
        self.cat2pe = cat2pe
        self.cat2kv = cat2kv
        assert len(set([self.cat2kv,self.cat2pe])) == 2,'must one True and another one False.'

        self.glob_param_outsize = glob_param_outsize
        self.imp_param_outsize = imp_param_outsize
        self.hyb_param_outsize = hyb_param_outsize
        self.src_hyb_param_outsize = src_hyb_param_outsize
        self.src_len = src_len
        self.rf_tgt_len = rf_tgt_len
        
        print(f"In params encoder layer:\n")
        print(f"{glob_param_outsize=}")
        print(f"{imp_param_outsize=}")
        print(f"{hyb_param_outsize=}")
#################################################################
        if self.cat2pe:
            self.model_out_size = glob_param_outsize + imp_param_outsize + hyb_param_outsize
            self.model_out_size_src = glob_param_outsize + imp_param_outsize + src_hyb_param_outsize

        elif self.cat2kv:
            self.model_out_size = glob_param_outsize + imp_param_outsize + hyb_param_outsize
            assert sep_glob_layer == False,'no need to have another layer'
            assert sep_imp_layer == False, 'no need to have another layer'
            self.rf_tgt_len,rf_tgt_len=src_len,src_len

            cat2kv_hyb_pool_methods = ['cat','minus','plus','add','+','-','subtraction','mean']
            self.cat2kv_hyb_pool = cat2kv_hyb_pool

            print(f"{self.cat2kv_hyb_pool=}")
            assert self.cat2kv_hyb_pool in cat2kv_hyb_pool_methods
            if self.cat2kv_hyb_pool == 'cat':
                self.model_out_size += src_hyb_param_outsize
            else:
                assert src_hyb_param_outsize==hyb_param_outsize, 'pool is not cat so, src_hyb_param_outsize==hyb_param_outsize'

        print(f"Dimension out should be: {self.model_out_size}")
        if self.model_out_size == 0:
            print(f"params encoder should have non zero out dim")
            print(50*'#')
            raise ValueError('params encoder should have non zero out dim')
        

##################################################################
        self.src_glob_params_encode_layer = None
        self.rf_tgt_glob_params_encode_layer = None
        if self.glob_param_outsize != 0:
            glob_param_layer_dict = dict(in_feature=glob_param_size,
                                            dim_hidden=dim_val,
                                            out_feature=glob_param_outsize,
                                            depth=depth,
                                            dropout=dropout_mlp)
            
            self.src_glob_params_encode_layer = Time_Step_Mlp(**glob_param_layer_dict,time_step=src_len)
            if sep_glob_layer: 
                self.rf_tgt_glob_params_encode_layer = Time_Step_Mlp(**glob_param_layer_dict,time_step=rf_tgt_len)
            
#################################################################

#################################################################
#Imp param MLP
        # imp_param = b,ts_f
        # imp_param_outsize = 32
        self.src_imp_params_encode_layer = None
        self.rf_tgt_imp_params_encode_layer = None

        if self.imp_param_outsize != 0:
            imp_param_layer_dict = dict(in_feature=imp_param_size,
                                        out_feature=imp_param_outsize,
                                        dim_hidden=dim_val,
                                        depth=depth,
                                        dropout=dropout_mlp)
            self.src_imp_params_encode_layer = Time_Step_Mlp(**imp_param_layer_dict,time_step=src_len)
            if sep_imp_layer:
                self.rf_tgt_imp_params_encode_layer = Time_Step_Mlp(**imp_param_layer_dict,time_step=rf_tgt_len)
#################################################################


#################################################################
#Hyb param SetTransformer
        self.src_hyb_params_encode_layer = None
        self.rf_tgt_hyb_params_encode_layer = None

        if not sep_hyb_layer:
            hyb_rf_tgt_len = src_len
        else:
            hyb_rf_tgt_len = rf_tgt_len
        print(f"As {sep_hyb_layer=} {hyb_rf_tgt_len=} {src_len=}")
        
        if hyb_use_transformer:
            set_transformer_dict = dict(dim_input=hyb_param_size,
                                            num_inds=32,
                                            dim_hidden=dim_val,
                                            num_heads=num_heads)
            
            if self.hyb_param_outsize !=0 :
                self.rf_tgt_hyb_params_encode_layer = SetTransformer(**set_transformer_dict,
                                                num_outputs=hyb_rf_tgt_len,
                                                dim_output=hyb_param_outsize)

            if self.src_hyb_param_outsize !=0 :
                if sep_hyb_layer:
                    self.src_hyb_params_encode_layer = SetTransformer(**set_transformer_dict,
                                                                    num_outputs=src_len,
                                                                    dim_output=src_hyb_param_outsize)
                else:
                    assert src_hyb_param_outsize == hyb_param_outsize
                    self.src_hyb_params_encode_layer = self.rf_tgt_hyb_params_encode_layer
                                    
        else :
            deep_set_dict = dict(dim_input=hyb_param_size,
                                    dim_hidden=dim_val,
                                    depth=depth,
                                    dropout=dropout_mlp
                                    )
            if self.hyb_param_outsize !=0 :
                self.rf_tgt_hyb_params_encode_layer = MeanPoolSet(**deep_set_dict,
                                                            dim_output = hyb_param_outsize,
                                                            num_outputs=hyb_rf_tgt_len
                                                            )
            if self.src_hyb_param_outsize !=0 :
                if sep_hyb_layer:
                    self.src_hyb_params_encode_layer = MeanPoolSet(**deep_set_dict,
                                                            dim_output=src_hyb_param_outsize,
                                                            num_outputs=src_len)
                else:
                    assert src_hyb_param_outsize == hyb_param_outsize
                    self.src_hyb_params_encode_layer = self.rf_tgt_hyb_params_encode_layer 
        
        print(50*"#")

    def forward(self,glob_param: Tensor, imp_param: Tensor,hyb_param: Tensor,src_hyb_param: Tensor):
        if self.DEBUG:
            print(50*"#")
            print(f"From From Encoded Params forward:\n{glob_param.size()=}\n{hyb_param.size()=}\n{src_hyb_param.size()=}")
        if self.cat2pe:
            encoded_param_src,encoded_param_tgt = self.forward_cat2pe(glob_param,imp_param,hyb_param,src_hyb_param)
            return encoded_param_src,encoded_param_tgt
        elif self.cat2kv:
            encoded_param_src = self.forward_cat2kv(glob_param,imp_param,hyb_param,src_hyb_param)
            return encoded_param_src

    def forward_cat2kv(self,glob_param: Tensor, imp_param: Tensor,hyb_param: Tensor,src_hyb_param: Tensor):

        if self.DEBUG: print(50*"#")

#####################################################################################################################
# param encoder
        if self.src_glob_params_encode_layer is not None:
            if self.DEBUG: print(f"From Encoded Params forward cat2kv: {glob_param.size()=}") 
            src_encoded_glob_param = self.src_glob_params_encode_layer(glob_param)
            if self.DEBUG: print(f"From Encoded Params forward cat2kv, after encoding: {src_encoded_glob_param.size()=}") 
        else:
            src_encoded_glob_param = None
            

        if self.src_imp_params_encode_layer is not None:
            if self.DEBUG: print(f"From Encoded Params forward cat2kv: {imp_param.size()=}")
            b,t,f = imp_param.size()
            assert t==1,'ts should be one for imp param'
            imp_param = imp_param.view(b,f)
            src_encoded_imp_param = self.src_imp_params_encode_layer(imp_param)
            if self.DEBUG: print(f"From Encoded Params forward cat2kv, after encoding: {src_encoded_imp_param.size()=}") 
        else:
            src_encoded_imp_param = None

        if self.src_hyb_params_encode_layer is not None or self.rf_tgt_hyb_params_encode_layer is not None: 

            if self.src_hyb_params_encode_layer is not None :

                if self.DEBUG: print(f"From Encoded Params forward cat2kv: {src_hyb_param.size()=}") 
                b,num_src,_,_ = src_hyb_param.size()
                src_hyb_param = rearrange(src_hyb_param,'b n d f -> (b n) d f')
                src_encoded_hyb_param = self.src_hyb_params_encode_layer(src_hyb_param)
                src_encoded_hyb_param =  rearrange(src_encoded_hyb_param,'(b n) d f -> b n d f',b=b,n=num_src)
                if self.DEBUG: print(f"From Encoded Params forward cat2kv, after encoding: {src_encoded_hyb_param.size()=}") 
                if len(src_encoded_hyb_param.size()) == 3: 
                    src_encoded_hyb_param=src_encoded_hyb_param.unsqueeze_(0)
                src_encoded_hyb_param=torch.mean(src_encoded_hyb_param,dim=1)
                src_encoded_hyb_param=src_encoded_hyb_param.view(b,self.src_len,self.src_hyb_param_outsize)

                if self.DEBUG: print(f"From Encoded Params forward cat2kv, after encoding: {src_encoded_hyb_param.size()=}") 
            else:
                src_encoded_hyb_param = None
            
            if self.rf_tgt_hyb_params_encode_layer is not None:
                if self.DEBUG: print(f"From Encoded Params forward cat2kv: {hyb_param.size()=}")
                b,n,ts,f=hyb_param.size()
                assert n ==1
                hyb_param = hyb_param.view(b,ts,f)
                rf_tgt_encoded_hyb_param = self.rf_tgt_hyb_params_encode_layer(hyb_param)
                rf_tgt_encoded_hyb_param=rf_tgt_encoded_hyb_param.view(b,self.rf_tgt_len,self.hyb_param_outsize)
                if self.DEBUG: print(f"From Encoded Params forward cat2kv, after encoding: {rf_tgt_encoded_hyb_param.size()=}") 
            else:
                rf_tgt_encoded_hyb_param = None

            
            if self.cat2kv_hyb_pool == 'cat':
                # merged_encoded_hyb = torch.cat((src_encoded_hyb_param,rf_tgt_encoded_hyb_param),dim=2)
                merged_encoded_hyb = self._none_include_cat(src_encoded_hyb_param,rf_tgt_encoded_hyb_param,dim=2)
            elif self.cat2kv_hyb_pool in ['minus','-','subtraction']:
                merged_encoded_hyb = rf_tgt_encoded_hyb_param-src_encoded_hyb_param
            elif self.cat2kv_hyb_pool in ['add','plus','+']:
                merged_encoded_hyb = rf_tgt_encoded_hyb_param+src_encoded_hyb_param
            elif self.cat2kv_hyb_pool in ['mean']:
                merged_encoded_hyb = (rf_tgt_encoded_hyb_param+src_encoded_hyb_param)/2.
            else:
                raise ValueError('no such cat2kv_hyb_pool method')

            if self.DEBUG: print(f"From Encoded Params forward cat2kv, after {self.cat2kv_hyb_pool=}: {merged_encoded_hyb.size()=}") 
        else:
            merged_encoded_hyb = None
            
        src_encoded_param = self._none_include_cat(src_encoded_glob_param,
                                                   src_encoded_imp_param,
                                                   merged_encoded_hyb)


        _,t,d = src_encoded_param.size()
        assert d == self.model_out_size,'dimention wrong'
        assert t == self.src_len

        if self.DEBUG: print(f"From Encoded Params forward cat2kv: {src_encoded_param.shape=}")
        if self.DEBUG: print(50*"#")

        return src_encoded_param



    def forward_cat2pe(self,glob_param: Tensor, imp_param: Tensor,hyb_param: Tensor,src_hyb_param: Tensor):
        
        if self.DEBUG: print(50*"#")

#####################################################################################################################
# param encoder
        if self.src_glob_params_encode_layer is not None:

            if self.DEBUG: print(f"From Encoded Params: {glob_param.size()=}") 
            src_encoded_glob_param = self.src_glob_params_encode_layer(glob_param)
            if self.DEBUG: print(f"From Encoded Params, after encoding: {src_encoded_glob_param.size()=}") 

            if self.rf_tgt_glob_params_encode_layer is None:
                rf_tgt_encoded_glob_param = src_encoded_glob_param[:,1:]
            else:
                rf_tgt_encoded_glob_param = self.rf_tgt_glob_params_encode_layer(glob_param)
            if self.DEBUG: print(f"From Encoded Params, after encoding: {rf_tgt_encoded_glob_param.size()=}") 

        else:
            src_encoded_glob_param = None
            rf_tgt_encoded_glob_param = None
            

        if self.src_imp_params_encode_layer is not None:
            if self.DEBUG: print(f"From Encoded Params: {imp_param.size()=}")
            b,t,f = imp_param.size()
            assert t==1,'ts should be one for imp param'
            imp_param = imp_param.view(b,f)
            src_encoded_imp_param = self.src_imp_params_encode_layer(imp_param)
            if self.DEBUG: print(f"From Encoded Params, after encoding: {src_encoded_imp_param.size()=}") 
            if self.rf_tgt_imp_params_encode_layer is None:
                rf_tgt_encoded_imp_param = src_encoded_imp_param[:,1:]
            else:
                rf_tgt_encoded_imp_param = self.rf_tgt_imp_params_encode_layer(imp_param)
            if self.DEBUG: print(f"From Encoded Params, after encoding: {rf_tgt_encoded_imp_param.size()=}") 

        else:
            src_encoded_imp_param = None
            rf_tgt_encoded_imp_param = None



        if self.src_hyb_params_encode_layer is not None :

            if self.DEBUG: print(f"From Encoded Params forward cat2pe: {src_hyb_param.size()=}") 
            b,num_src,_,_ = src_hyb_param.size()
            src_hyb_param = rearrange(src_hyb_param,'b n d f -> (b n) d f')
            src_encoded_hyb_param = self.src_hyb_params_encode_layer(src_hyb_param)
            src_encoded_hyb_param =  rearrange(src_encoded_hyb_param,'(b n) d f -> b n d f',b=b,n=num_src)
            if self.DEBUG: print(f"From Encoded Params forward cat2pe, after encoding: {src_encoded_hyb_param.size()=}") 
            if len(src_encoded_hyb_param.size()) == 3: 
                src_encoded_hyb_param=src_encoded_hyb_param.unsqueeze_(0)
            src_encoded_hyb_param=torch.mean(src_encoded_hyb_param,dim=1)
            src_encoded_hyb_param=src_encoded_hyb_param.view(b,self.src_len,self.src_hyb_param_outsize)

            if self.DEBUG: print(f"From Encoded Params forward cat2pe, after encoding: {src_encoded_hyb_param.size()=}") 
        else:
            src_encoded_hyb_param = None
        
        if self.rf_tgt_hyb_params_encode_layer is not None:
            if self.DEBUG: print(f"From Encoded Params forward cat2pe: {hyb_param.size()=}")
            b,n,ts,f=hyb_param.size()
            assert n ==1
            hyb_param = hyb_param.view(b,ts,f)
            rf_tgt_encoded_hyb_param = self.rf_tgt_hyb_params_encode_layer(hyb_param)
            if self.DEBUG: print(f"From Encoded Params forward cat2pe: {rf_tgt_encoded_hyb_param.size()=}")
            b,ts,f=rf_tgt_encoded_hyb_param.size()

            if ts == self.rf_tgt_len:
                pass
            elif ts-1 == self.rf_tgt_len:
                rf_tgt_encoded_hyb_param = rf_tgt_encoded_hyb_param[:,1:]
            else:
                raise ValueError('dimension wrong')

            rf_tgt_encoded_hyb_param=rf_tgt_encoded_hyb_param.view(b,self.rf_tgt_len,self.hyb_param_outsize)
            if self.DEBUG: print(f"From Encoded Params forward cat2pe, after encoding: {rf_tgt_encoded_hyb_param.size()=}") 
        else:
            rf_tgt_encoded_hyb_param = None


        src_encoded_param = self._none_include_cat(src_encoded_glob_param,
                                                   src_encoded_imp_param,
                                                   src_encoded_hyb_param)

        rf_tgt_encoded_param = self._none_include_cat(rf_tgt_encoded_glob_param,
                                                    rf_tgt_encoded_imp_param,
                                                    rf_tgt_encoded_hyb_param)
        _,t,d = src_encoded_param.size()
        assert d == self.model_out_size,'dimention wrong'
        assert t == self.src_len
        _,t,d = rf_tgt_encoded_param.size()
        assert d == self.model_out_size,'dimention wrong'
        assert t == self.rf_tgt_len

        if self.DEBUG: print(f"From Encoded Params: {src_encoded_param.shape=}")
        if self.DEBUG: print(f"From Encoded Params: {rf_tgt_encoded_param.shape=}")

        if self.DEBUG: print(50*"#")

        return src_encoded_param,rf_tgt_encoded_param
    
    def _none_include_cat(self,*args,dim=2):
        out = None
        for item in args:
            if item is not None:
                if out is not None:
                    out = torch.cat((out,item),dim=dim)
                else:
                    out = item
        return out

    @staticmethod
    def cat_param2src(encoded_src:torch.Tensor,encoded_param:torch.Tensor,wn_s:int=None,wn_e:int=None)->torch.Tensor:
        if wn_s is None:
            param_subarrays = encoded_param
        else:
            param_subarrays = encoded_param[:,wn_s:wn_e,:]

        encoded_src = torch.cat((encoded_src,param_subarrays),dim=2)
        return encoded_src