import torch
from torch import nn
from torch import Tensor
from . import positional_encoder as pe
from . import utils
from .layers import MLP_PositionwiseFeedForward,PositionwiseFeedForward
from .transformer_layers import TransforemerEncoder,TransforemerDecoder
from .pytorch_transformer import TransformerEncoder_Pytorch,TransformerDecoder_Pytorch
from .error_correction_encoded_params import Error_Correction_Encoded_Param_Layer
from ML_dmft.utility.tools import get_bool_env_var


class Transformer(nn.Module):
    def __init__(self, 
        input_seq_size: int,
        input_seq_len : int,
        dec_seq_len: int,
        tot_seq_len:int,
        glob_param_size:int,
        imp_param_size:int,
        hyb_param_size:int,
        tail_seq_len:int,
        beta:int,
        glob_param_outsize:int=0,
        imp_param_outsize:int=0,
        hyb_param_outsize:int=0,
        src_hyb_param_outsize:int=0,
        cat2kv_hyb_pool:str='cat',
        sep_glob_layer:bool=False,
        sep_imp_layer:bool=False,
        sep_hyb_layer:bool=False,
        param_cat_method:int=2,
        seq_pos_method:str="3",
        dim_val: int=512,
        n_encoder_layers: int=4,
        n_decoder_layers: int=4,
        n_heads: int=4,
        depth_enc_param_layer : int = 0,
        depth_input_layer : int =0,
        dropout_enc_param_layer:float=0.,
        droupout_input_layer:float=0,
        dropout_encoder: float=0., 
        dropout_decoder: float=0.,
        dropout_pos_enc: float=0.,
        dim_feedforward_encoder: int=2048,
        dim_feedforward_decoder: int=2048,
        num_predicted_features: int=1,
        tgt_src_mask_off = False,
        enc_dec_input_sep= False,
        predict_unknown_only=False,
        tail_from_file = False,
        hyb_use_transformer=False,
        tgt_mask_lookbackward_len=100,
        src_mask_lookbackward_len=100,
        src_mask_lookforward_len=0,
        decoder_positioanl_norm=False,
        ): 
    

        """
        Args:
            input_seq_size: int, number of input variables. 1 if univariate.
            dec_seq_len: int, the length of the input sequence fed to the decoder
            dim_val: int, aka d_model. All sub-layers in the model produce 
                     outputs of dimension dim_val
            n_encoder_layers: int, number of stacked encoder layers in the encoder
            n_decoder_layers: int, number of stacked encoder layers in the decoder
            n_heads: int, the number of attention heads (aka parallel attention layers)
            dropout_encoder: float, the dropout rate of the encoder
            dropout_decoder: float, the dropout rate of the decoder
            dropout_pos_enc: float, the dropout rate of the positional encoder
            dim_feedforward_encoder: int, number of neurons in the linear layer 
                                     of the encoder
            dim_feedforward_decoder: int, number of neurons in the linear layer 
                                     of the decoder
            num_predicted_features: int, the number of features you want to predict.
            hyb_use_transformer : bool, options to switch between 
                                    set-transformer and set.
            tgt_src_mask_off : bool, options to turn off tgt_src_mask.
            tgt_mask_lookbackward_len : int, look back length.
            src_mask_lookbackward_len : int, look back length src_tgt.
            src_mask_lookforward_len : 
            num_points2pred : int, number of data by transformer.
        """
        super().__init__()

        print(50*'#')
        print(50*'#')
        print(50*'#','\n')
        print("Two Solver Transfromer Init:")
        self.DEBUG=get_bool_env_var('ML_dmft_DEBUG')

        num_approx_solutions = input_seq_size

        print(f"{num_approx_solutions=}")

        pytorch_tranformer = True

        self.tgt_src_mask_off = tgt_src_mask_off
        self.predict_unknown_only = predict_unknown_only
        self.tail_from_file = tail_from_file
        self.decoder_positioanl_norm = decoder_positioanl_norm

        assert input_seq_len == tot_seq_len, 'two sequence should have same length.'
        
        self.input_seq_len = input_seq_len
        self.dec_seq_len = dec_seq_len
        self.tot_seq_len = tot_seq_len #try to not use this variable.
        self.tail_seq_len = tail_seq_len

        self.num_points2pred = self.dec_seq_len-self.tail_seq_len
        assert self.num_points2pred >=0
        print(f"{self.num_points2pred=} {self.tail_seq_len=}") 

        self.tgt_mask_lookbackward_len = tgt_mask_lookbackward_len
        self.src_mask_lookbackward_len = src_mask_lookbackward_len
        self.src_mask_lookforward_len = src_mask_lookforward_len

        self.beta = beta
        _sum_encoded_param_dim = glob_param_outsize + imp_param_outsize + hyb_param_outsize + src_hyb_param_outsize

#####################################################################################
        self.cat2pe,self.cat2kv,self.tst = False, False, False

        if param_cat_method == 0:
            print(f'{param_cat_method=} cat2kv')
            self.cat2kv = True
            assert _sum_encoded_param_dim != 0
        elif param_cat_method == 1:
            print(f'{param_cat_method=} cat2pe')
            self.cat2pe = True
            assert _sum_encoded_param_dim != 0
        elif param_cat_method == 2:
            self.tst = True
            assert _sum_encoded_param_dim == 0, 'Time-serial Transformer Mode should have encoded_param_dim zero'
            assert depth_enc_param_layer ==0, 'Time-serial depth_enc_param_layer should be zero'
            print(f'{param_cat_method=} tst')
        else:
            raise ValueError('param_cat_method no such options')

        assert len(set([self.cat2kv,self.cat2pe,self.tst])) == 2,'must one True and another one False.'
#####################################################################################

#####################################################################################
#Seq Pos Encode

        PE=pe.Pos_Encoder(str(seq_pos_method),self.beta,
                    tot_seq_len,input_seq_len,dec_seq_len,
                    dim_val,dropout_pos_enc)

        input_seq_size+= PE.PE_OUT_SEQ_SIZE
        self.src_positional_encoding_layer = PE.src_positional_encoding_layer
        self.tgt_positional_encoding_layer = PE.tgt_positional_encoding_layer
        self.PE_LAYERS_FIRST = PE.PE_LAYERS_FIRST 


        output_seq_size=1+PE.PE_OUT_SEQ_SIZE
    
#####################################################################################


################################# encoded_param_layer ###############################
        if self.tst:
            self.encoded_param_layer = None
        else:
            self.encoded_param_layer=Error_Correction_Encoded_Param_Layer(
                            glob_param_size=glob_param_size,
                            imp_param_size=imp_param_size,
                            hyb_param_size=hyb_param_size,
                            src_len=self.input_seq_len,
                            rf_tgt_len=self.dec_seq_len,
                            dim_val=dim_val,
                            depth=depth_enc_param_layer,
                            dropout_mlp=dropout_enc_param_layer,
                            glob_param_outsize = glob_param_outsize,
                            imp_param_outsize = imp_param_outsize,
                            hyb_param_outsize = hyb_param_outsize,
                            src_hyb_param_outsize = src_hyb_param_outsize,
                            hyb_use_transformer = hyb_use_transformer,
                            cat2pe = self.cat2pe,
                            cat2kv = self.cat2kv,
                            cat2kv_hyb_pool = cat2kv_hyb_pool,
                            sep_glob_layer = sep_glob_layer,
                            sep_imp_layer = sep_imp_layer,
                            sep_hyb_layer = sep_hyb_layer)
#####################################################################################

#####################################################################################
        if self.cat2kv:
            print(f"{self.cat2kv=}")
            decoder_kv_in_dim = self.encoded_param_layer.model_out_size + dim_val
            self.decoder_kv_layer = PositionwiseFeedForward(decoder_kv_in_dim,dim_val)
            
        elif self.cat2pe:
            input_seq_size +=  self.encoded_param_layer.model_out_size_src
            output_seq_size += self.encoded_param_layer.model_out_size
            print(f"self.cat2pe: {input_seq_size=} {output_seq_size=}")
#####################################################################################


#####################################################################################
        print(f"encoder_input_layer {input_seq_size=}")
        self.encoder_input_layer = MLP_PositionwiseFeedForward(
                                                in_feature=input_seq_size,
                                                hidden_feature=dim_val,
                                                out_feature=dim_val,
                                                drop_prob=droupout_input_layer,
                                                depth=depth_input_layer)


        if enc_dec_input_sep:
            print(f"decoder_input_layer {output_seq_size=}")
            self.decoder_input_layer = MLP_PositionwiseFeedForward(
                                                in_feature=output_seq_size,
                                                hidden_feature=dim_val,
                                                out_feature=dim_val,
                                                drop_prob=droupout_input_layer,
                                                depth=depth_input_layer)
        else:
            if input_seq_size != output_seq_size: 
                raise ValueError('Encoder input layer and Decoder input layer dimenison are different')
            self.decoder_input_layer = self.encoder_input_layer
#####################################################################################
        
        if not pytorch_tranformer:
            self.transformer_encoder = TransforemerEncoder(
                                    depth=n_encoder_layers,
                                    d_model=dim_val, 
                                    n_head=n_heads,
                                    ffn_hidden=dim_feedforward_encoder,
                                    drop_prob=dropout_encoder,
                                    norm=nn.LayerNorm, 
                                    act_layer=nn.GELU
                                    )
            
            self.transformer_decoder = TransforemerDecoder(d_model=dim_val,
                                            num_predicted_features=num_predicted_features,
                                            ffn_hidden=dim_feedforward_decoder,
                                            n_head=n_heads,
                                            n_decoder_layers=n_decoder_layers,
                                            drop_prob=dropout_decoder,
                                            norm=nn.LayerNorm,
                                            act_layer=nn.GELU
                                            )
        else:
            self.transformer_encoder = TransformerEncoder_Pytorch(d_model=dim_val,
                                                    ffn_hidden=dim_feedforward_encoder,
                                                    n_head=n_heads,
                                                    n_encoder_layers=n_encoder_layers,
                                                    drop_prob=dropout_encoder+1e-16)

            self.transformer_decoder = TransformerDecoder_Pytorch(d_model=dim_val,
                                            num_predicted_features=num_predicted_features,
                                            ffn_hidden=dim_feedforward_decoder,
                                            n_head=n_heads,
                                            n_decoder_layers=n_decoder_layers,
                                            drop_prob=dropout_decoder+1e-16)
#####################################################################################
        print('\n')
        print(50*'#')
        print(50*'#')
        print(50*'#')


    def forward(self,glob_param: Tensor, imp_param: Tensor,hyb_param: Tensor,src_hyb_param: Tensor,
                src: Tensor, tgt_tail: Tensor,rf_tgt: Tensor = None) -> Tensor:
        r"""
        
        """
        if self.DEBUG: print(50*"#")
        assert src.size()[1]==self.tot_seq_len

        #####################################################################################################################
        # param encoder
        src_encoded_param = None
        rf_tgt_encoded_param = None

        if self.encoded_param_layer is not None:
            if self.cat2pe:
                src_encoded_param,rf_tgt_encoded_param = self.encoded_param_layer(glob_param,imp_param,hyb_param,src_hyb_param)
                if self.DEBUG: print(f"From model.forward(): {src_encoded_param.size()=} {rf_tgt_encoded_param.size()=}")

            elif self.cat2kv:
                src_encoded_param = self.encoded_param_layer(glob_param,imp_param,hyb_param,src_hyb_param)
                if self.DEBUG: print(f"From model.forward(): {src_encoded_param.size()=}")
        #####################################################################################################################

        if self.DEBUG: print(f"From model.forward(): Size of src as given to forward(): {src.size()=}")

        if self.DEBUG and rf_tgt is not None: print("From model.forward(): rf_tgt size = {}".format(rf_tgt.size()))

        encoded_src = src

        # drop 
        if src.size()[2] != 1:
            src = src[:,:,0].unsqueeze(-1)
            if self.DEBUG : print(f"From model.forward(): After drop the rest {src.size()=}")


        if self.PE_LAYERS_FIRST:
            encoded_src = self.src_positional_encoding_layer(encoded_src) # src shape: [batch_size, src length, dim_val] regardless of number of input features
            if self.DEBUG: print("From model.forward(): Size of src after pos_enc layer: {}".format(encoded_src.size()))

        if self.cat2pe:

            encoded_src=Error_Correction_Encoded_Param_Layer.cat_param2src(encoded_src,src_encoded_param)
            if self.DEBUG: print(f"From model.forward(),self.cat2pe, after cat {encoded_src.size()=}")

        encoded_src = self.encoder_input_layer(encoded_src)
        if self.DEBUG: print("From model.forward(): Size of src after input layer: {}".format(encoded_src.size()))

        if not self.PE_LAYERS_FIRST:
            encoded_src = self.src_positional_encoding_layer(encoded_src) # src shape: [batch_size, src length, dim_val] regardless of number of input features
            if self.DEBUG: print("From model.forward(): Size of src after pos_enc layer: {}".format(encoded_src.size()))

        encoded_src = self.transformer_encoder(encoded_src)
        if self.DEBUG: print("From model.forward(): Size of src after encoder layer: {}".format(encoded_src.size()))

        if self.cat2kv:
            encoded_src=Error_Correction_Encoded_Param_Layer.cat_param2src(encoded_src,src_encoded_param)
            encoded_src = self.decoder_kv_layer(encoded_src)
            if self.DEBUG: print(f"From model.forward(),self.cat2kv, after cat {encoded_src.size()=}")

        if rf_tgt is None:
            rf_tgt=self._predict(src,encoded_src,rf_tgt_encoded_param,tgt_tail)
            if self.DEBUG: print(f"From model.forward(): after predict {rf_tgt.shape=}")
            decoder_output=self._forward_decode(src,encoded_src,rf_tgt,rf_tgt_encoded_param)
            if self.tail_from_file: decoder_output[:,:self.tail_seq_len]=tgt_tail    

        else:
            decoder_output=self._forward_decode(src,encoded_src,rf_tgt,rf_tgt_encoded_param)

        if self.DEBUG: print(f"From model.forward(): {encoded_src.shape=} {rf_tgt.shape=}")
        if self.DEBUG: print(50*"#")
        return decoder_output
    
    def _forward_decode(self,src:torch.Tensor,encoded_src:torch.Tensor,
                        rf_tgt:torch.Tensor,rf_tgt_encoded_param:torch.Tensor=None)->torch.Tensor:

        src_mask,tgt_mask = self._gen_subsequent_mask(encoded_src,rf_tgt)
                
        if self.DEBUG: print(f"In forward decdode: \n{encoded_src.shape= } {rf_tgt.shape=} {src_mask.shape=} \n{tgt_mask.shape=}  {self.dec_seq_len=}")
        if self.tgt_src_mask_off: src_mask = None

        ##################################################

        trg_wn_e = self.dec_seq_len + 1 #dec_seq_len= 32 
        trg_wn_s = self.dec_seq_len - rf_tgt.shape[1] + 1 #32

        ##################################################

        decoder_output = rf_tgt
        if self.decoder_positioanl_norm:
            if self.DEBUG: print(f"In forward decdode: {self.decoder_positioanl_norm=}")
            if self.DEBUG: print(f"In forward decdode: {decoder_output.size()=} {src.size()=}")
            if self.DEBUG: print(f"In forward decdode: {src[0,:]=}")
            if self.DEBUG: print(f"In forward decdode: {decoder_output[0,:]=}")
            decoder_output = decoder_output - src[:,:decoder_output.size()[1]]
            if self.DEBUG: print(f"In forward decdode, after substract src: {decoder_output[0,:]=}")


        if self.PE_LAYERS_FIRST:
            if self.DEBUG: print(f"In forward decdode: {trg_wn_s=} {trg_wn_e=}")
            decoder_output = self.tgt_positional_encoding_layer(decoder_output,trg_wn_s,trg_wn_e)

        if self.DEBUG: print("In forward decdode: Size of decoder_output after positional encoding: {}".format(decoder_output.size()))
        ##################################################
        if self.cat2pe:
            _wn_e = self.dec_seq_len
            _wn_s = self.dec_seq_len - decoder_output.shape[1] 
            decoder_output = Error_Correction_Encoded_Param_Layer.cat_param2src(decoder_output,rf_tgt_encoded_param,_wn_s,_wn_e)
            if self.DEBUG: print(f"From model.forward(), after cat {decoder_output.size()}")
        ##################################################

        # Pass decoder input through decoder input layer
        if self.DEBUG: print(f"In forward decdode: before decoder_input_layer {decoder_output.size()=}")
        decoder_output = self.decoder_input_layer(decoder_output) # src shape: [target sequence length, batch_size, dim_val] regardless of number of input features
        if self.DEBUG: print("In forward decdode: Size of decoder_output after linear decoder layer: {}".format(decoder_output.size()))

        if not self.PE_LAYERS_FIRST:
            if self.DEBUG: print(f"In forward decdode: {trg_wn_s=} {trg_wn_e=}")
            decoder_output = self.tgt_positional_encoding_layer(decoder_output,trg_wn_s,trg_wn_e)

        if self.DEBUG: print(f"In forward decdode, before transformer decoder: {decoder_output.size()=}\n{encoded_src.size()=}\n")
        decoder_output = self.transformer_decoder(trg=decoder_output, enc_src=encoded_src, 
                                                    trg_mask=tgt_mask, src_mask=src_mask)
        if self.DEBUG: print("In forward decdode: Size of decoder_output after decoder: {}".format(decoder_output.size()))
        
        if self.decoder_positioanl_norm:
            if self.DEBUG: print(f"In forward decdode: {self.decoder_positioanl_norm=}")
            if self.DEBUG: print(f"In forward decdode: {decoder_output.size()=} {src.size()=}")
            if self.DEBUG: print(f"In forward decdode: {src[0,:]=}")
            if self.DEBUG: print(f"In forward decdode: {decoder_output[0,:]=}")
            decoder_output += src[:,1:decoder_output.size()[1]+1]
            if self.DEBUG: print(f"In forward decdode, after add src: {decoder_output[0,:]=}")
        
        return decoder_output
    
    def _predict(self,src:torch.Tensor,
                    encoded_src:torch.Tensor,
                    rf_tgt_encoded_param:torch.Tensor,
                    tgt_tail:torch.Tensor)->torch.Tensor:
        """
        Principle : anything we know. we put it into tgt.
        """

        if self.DEBUG: print(f"{50*'#'} Predict")
        if self.DEBUG: print(f"From Predict: {len(src)=}")

        if self.DEBUG : print(f"From Predict: {src[0,1]=} {tgt_tail[0,0]=} {tgt_tail.size()=}")
        rf_tgt = src[:,0,:]
        rf_tgt = rf_tgt.unsqueeze(-1) if len(rf_tgt.size()) == 2 else rf_tgt
        # print(f"From Predict: {rf_tgt.shape=}")
        #HERE to verify if dimention is correct.

        if self.predict_unknown_only: 
            if self.DEBUG: print(f"From Predict: {self.dec_seq_len-1=} {self.tail_seq_len}")
            rf_tgt = torch.cat((rf_tgt,tgt_tail[:,:min(self.dec_seq_len-1,self.tail_seq_len)]),dim=1) #tgt_tail
            if self.DEBUG : print(f"From Predict: {self.predict_unknown_only=}")
            if self.DEBUG : print(f"From Predict: {rf_tgt.size()=}")
            if self.DEBUG : print(f"From Predict: {src[0,1]=} {tgt_tail[0,0]=} {tgt_tail.size()}")

        if self.DEBUG : print(f"From Predict: {rf_tgt.size()=}")
        if self.DEBUG : print(f"From Predict: {self.num_points2pred =} {self.dec_seq_len-rf_tgt.size()[1]=}")
        
        for idx in range(self.dec_seq_len-rf_tgt.size()[1]):
            if self.DEBUG: print(f"From Predict: {idx=} {rf_tgt[0,:,:].T=}")

            decoder_output=self._forward_decode(src,encoded_src,rf_tgt,rf_tgt_encoded_param)
            last_pred_val = decoder_output[:, -1, :] #everything predicted by decoder.
            last_pred_val = last_pred_val.unsqueeze(-1) if len(last_pred_val.size()) == 2 else last_pred_val
            self.DEBUG: print(f"From Predict: {rf_tgt.shape=} {last_pred_val.shape=}")
            rf_tgt = torch.cat((rf_tgt, last_pred_val), 1)

        if self.DEBUG: print(f"{50*'#'} predict")
        return rf_tgt
    
    
    def _gen_subsequent_mask(self,src:torch.Tensor,tgt:torch.Tensor)->tuple([torch.Tensor,torch.Tensor]):
        dim_a = tgt.shape[1]

        tgt_mask = utils.lookbackward_square_mask(dim_a,self.tgt_mask_lookbackward_len).to(tgt.device)

        src_tgt_mask = utils.gen_make_bidirectional_mask_rf_tgt_src(dim_a,
                                        lookbackward_len=self.src_mask_lookbackward_len,
                                        lookforward_len=self.src_mask_lookforward_len,
                                        input_seq_len=self.input_seq_len,
                                        dec_seq_len=self.dec_seq_len,
                                        tot_seq_len=self.tot_seq_len).to(tgt.device)      

        
        if self.DEBUG: print(f"{tgt_mask=}")
        if self.DEBUG: print(f"{src_tgt_mask=}")

        return src_tgt_mask,tgt_mask
        
        