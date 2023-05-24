from .cosine_pos import PositionalEncoder as Cosine_PE
from .learned_pos import Learned_PositionalEncoder_Concatenate,Learned_PositionalEncoder
from .matsu_pos import Matsubara_PositionalEncoder_Concatenate
from .dev import Dev_PE


class Pos_Encoder():
    def __init__(self,seq_pos_method:str,beta:int,
                        tot_seq_len:int,input_seq_len:int,dec_seq_len:int,
                        dim_val:int,
                        dropout_pos_enc:float,
                        ) -> None:
        super().__init__()

        """
        Args:
            seq_pos_method: str,0,1,2,3 corresponding to each method
            beta: int, temperature (1/eV)
            tot_seq_len: int, length of total seq.
            input_seq_len:
            dec_seq_len:
            dim_val: dim value of encoding layer
            dropout_pos_enc: potional layer dropout
            PE_OUT_SEQ_SIZE: input seq dim, b,t,d = inputs.shape d is the input seq size.
        """

        series_pos_embedding_method_list = ['cosine','learned_pos_addition','learned_pos_concatnate','matsu_pos_concatnate','dev']
        # seq_pos_method = str(seq_pos_method)
        seq_pos_embedding_method = series_pos_embedding_method_list[int(float(seq_pos_method))]

        print(f"{seq_pos_method=}: {seq_pos_embedding_method=}")

        src_wn_e = tot_seq_len # 32 
        src_wn_s = tot_seq_len - input_seq_len # 32-16
        print(f"{src_wn_s=} {src_wn_e=}") 

        PE_OUT_SEQ_SIZE = 0

        if seq_pos_embedding_method=='dev':
            self.src_positional_encoding_layer = Dev_PE(wn_s=src_wn_s,
                                                        wn_e=src_wn_e,
                                                        max_len=tot_seq_len,
                                                        beta=beta,
                                                        out_feature=2,
                                                        )
            
            self.tgt_positional_encoding_layer = self.src_positional_encoding_layer

            self.PE_LAYERS_FIRST = True
            PE_OUT_SEQ_SIZE = 0
            raise AssertionError('DEV MODE')


        elif seq_pos_embedding_method=='cosine':

            self.src_positional_encoding_layer = Cosine_PE(
                                                d_model=dim_val,
                                                dropout=dropout_pos_enc,
                                                max_seq_len=tot_seq_len,
                                                wn_s=src_wn_s,
                                                wn_e=src_wn_e
                                                )
            
            self.tgt_positional_encoding_layer = Cosine_PE(
                                    d_model=dim_val,
                                    dropout=dropout_pos_enc,
                                    max_seq_len=tot_seq_len,
                                    )

            self.PE_LAYERS_FIRST = False

        
        elif seq_pos_embedding_method=='learned_pos_addition':
            
            self.src_positional_encoding_layer = Learned_PositionalEncoder(wn_s=src_wn_s,
                                                                wn_e=src_wn_e,
                                                                max_len=tot_seq_len)
            
            self.tgt_positional_encoding_layer = self.src_positional_encoding_layer
            self.PE_LAYERS_FIRST = True
        
            
        elif seq_pos_embedding_method == 'learned_pos_concatnate':

            dim_pos=1
            self.pe_encoder_layer = Learned_PositionalEncoder_Concatenate(wn_s=src_wn_s,
                                                                            wn_e=src_wn_e,
                                                                            max_len=tot_seq_len,
                                                                            dim_pos=dim_pos
                                                                            )
            self.src_positional_encoding_layer = self.pe_encoder_layer
            self.tgt_positional_encoding_layer = self.pe_encoder_layer

            PE_OUT_SEQ_SIZE += dim_pos
            self.PE_LAYERS_FIRST = True


        elif seq_pos_embedding_method == 'matsu_pos_concatnate':

            print(f"model {beta=}")

            if seq_pos_method == "3" or seq_pos_method == "3.0":
                transform_method='None'
        
            elif seq_pos_method == "3.1":
                transform_method='normalised'

            elif seq_pos_method == "3.2":
                transform_method='standardised'
            else:
                raise AssertionError('no such transform method.')

            self.src_positional_encoding_layer = Matsubara_PositionalEncoder_Concatenate(beta=beta,
                                                                                        wn_s=src_wn_s,
                                                                                        wn_e=src_wn_e,
                                                                                        max_len=tot_seq_len,                                  
                                                                                        transform_method=transform_method,
                                                                                        )

            self.tgt_positional_encoding_layer = Matsubara_PositionalEncoder_Concatenate(beta=beta,
                                                                                        max_len=tot_seq_len,
                                                                                        transform_method=transform_method,
                                                                                        )

            PE_OUT_SEQ_SIZE += 1
            self.PE_LAYERS_FIRST = True
        
        Pos_Encoder.help_info(seq_pos_method,self.PE_LAYERS_FIRST,PE_OUT_SEQ_SIZE,{"beta":beta})

        self.PE_OUT_SEQ_SIZE = PE_OUT_SEQ_SIZE

    @staticmethod
    def help_info(seq_pos_method:str=None,PE_LAYERS_FIRST:bool=None,PE_OUT_SEQ_SIZE:int=None,args:dict=None):
        long_text=f"""
{50*'#'}
Positional Encoder Interface

Options
-------
0: Cosine
1: Learned_Pos_addition
2: Learned_PositionalEncoder_Concatenate
3: Matsu_Pos_Concatnate
    3.0 Raw matsu
    3.1 Normalised
    3.2 Standardised

{seq_pos_method=} is Choosen
{PE_LAYERS_FIRST=} 
{PE_OUT_SEQ_SIZE=}
{args}
{50*'#'}
        """
        print(long_text)