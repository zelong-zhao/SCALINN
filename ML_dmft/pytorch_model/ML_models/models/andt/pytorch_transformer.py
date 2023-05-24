from torch import nn

class TransformerDecoder_Pytorch(nn.Module):
    def __init__(self,d_model, num_predicted_features, ffn_hidden, n_head, n_decoder_layers, drop_prob) -> None:
        super().__init__()
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=ffn_hidden,
            activation=nn.GELU(),
            dropout=drop_prob,
            batch_first=True
            )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers, 
            norm=None
            )
    
        self.linear_mapping = nn.Linear(
            in_features=d_model, 
            out_features=num_predicted_features
            )
    def forward(self, trg, enc_src, trg_mask, src_mask):

        trg = self.decoder(trg, enc_src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear_mapping(trg)
        return output

class TransformerEncoder_Pytorch(nn.Module):
    def __init__(self,d_model, ffn_hidden, n_head, n_encoder_layers, drop_prob) -> None:
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_head,
            dim_feedforward=ffn_hidden,
            activation=nn.GELU(),
            dropout=drop_prob,
            batch_first=True
            )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers, 
            norm=None
            )

    def forward(self, src):

        output = self.encoder(src)

        return output