import torch
import torch.nn as nn 
from torch import nn, Tensor

class Pos_Encoder():
    def __init__(self,seq_pos_method:int,beta:int,
                        tot_seq_len:int,input_seq_len:int,dec_seq_len:int,
                        dim_val:int,
                        dropout_pos_enc:float,
                        input_seq_size:int,
                        ) -> None:
        super().__init__()

        """
        Args:
            seq_pos_method: int,0,1,2,3 corresponding to each method
            beta: int, temperature (1/eV)
            tot_seq_len: int, length of total seq.
            input_seq_len:
            dec_seq_len:
            dim_val: dim value of encoding layer
            dropout_pos_enc: potional layer dropout
            input_seq_size: input seq dim, b,t,d = inputs.shape d is the input seq size.
        """

        series_pos_embedding_method_list = ['cosine','learned_pos_addition','learned_pos_concatnate','matsu_pos_concatnate']
        seq_pos_embedding_method = series_pos_embedding_method_list[seq_pos_method]
        print(f"{seq_pos_method=}: {seq_pos_embedding_method=}")

        if seq_pos_embedding_method=='learned_pos_addition':
            self.src_positional_encoding_layer = Learned_PositionalEncoder(
                                                input_seq_len,dim_val
                                                )
            
            self.tgt_positional_encoding_layer = PositionalEncoder(
                                    d_model=dim_val,
                                    dropout=dropout_pos_enc,
                                    max_seq_len=dec_seq_len,
                                    batch_first=True,
                                    )
            
        elif seq_pos_embedding_method=='cosine':
            self.src_positional_encoding_layer = PositionalEncoder(
                                                d_model=dim_val,
                                                dropout=dropout_pos_enc,
                                                max_seq_len=input_seq_len,
                                                batch_first=True,
                                                )
            
            self.tgt_positional_encoding_layer = PositionalEncoder(
                                    d_model=dim_val,
                                    dropout=dropout_pos_enc,
                                    max_seq_len=dec_seq_len,
                                    batch_first=True,
                                    )
            
        elif seq_pos_embedding_method == 'learned_pos_concatnate':
            self.src_positional_encoding_layer = Learned_PositionalEncoder_Concatenate(input_seq_len)
            self.tgt_positional_encoding_layer = Learned_PositionalEncoder_Concatenate(dec_seq_len)
            input_seq_size += 1
        
        elif seq_pos_embedding_method == 'matsu_pos_concatnate':

            print(f"model {beta=}")
            src_wn_e = tot_seq_len # 32 
            src_wn_s = tot_seq_len - input_seq_len # 32-16
            print(f"{src_wn_s=} {src_wn_e=}")
            self.src_positional_encoding_layer = Matsubara_PositionalEncoder_Concatenate(beta=beta,
                                                                                        wn_s=src_wn_s,
                                                                                        wn_e=src_wn_e,
                                                                                        max_len=tot_seq_len,
                                                                                        )

            tgt_wn_s = 1 
            tgt_wn_e = tgt_wn_s + dec_seq_len # 1+16

            print(f"{tgt_wn_s=} {tgt_wn_e=}")
            self.tgt_positional_encoding_layer = Matsubara_PositionalEncoder_Concatenate(beta=beta,
                                                                                        wn_s=tgt_wn_s,
                                                                                        wn_e=tgt_wn_e,
                                                                                        max_len=tot_seq_len
                                                                                        )

            # self.tgt_pe_encoder_in_decoder_layer = 

            input_seq_size += 1


        self.input_seq_size = input_seq_size