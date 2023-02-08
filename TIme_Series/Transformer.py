#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                                   # pe = [max_len,model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # position = [max_len,1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))   # 전체 길이의 절반만큼의 exp
        pe[:, 0::2] = torch.sin(position * div_term)                                                 # 짝수에는 sin
        pe[:, 1::2] = torch.cos(position * div_term)                                                 # 홀수에는 cos
        pe = pe.unsqueeze(0).transpose(0, 1)                                                         # pe = [max_len , 1, model]
        self.register_buffer('pe', pe)                                                               # pe matrix는 업데이트 되지 않는다

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]                                                               # x에 positional value 더해서 인코딩
        return x
    
class TFModel(nn.Module):
    def __init__(self,variable,inp_seq,out_seq,d_model, nhead, nlayers, dropout=0.5):
        super(TFModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers) 
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,dropout=dropout,batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=nlayers)

        self.encoder = nn.Sequential(
            nn.Linear(variable, d_model//2),
            nn.Linear(d_model//2, d_model)
        )

        self.decoder = nn.Sequential(
            nn.Linear(1, d_model//2),
            nn.Linear(d_model//2, d_model)
        )

        self.fc_layer =  nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.Linear(d_model//2, 1)
        )
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt,tgt_mask):
        tgt = tgt.unsqueeze(2)

        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        tgt = self.decoder(tgt)
        tgt = self.pos_encoder(tgt)
        result = self.transformer_decoder(tgt,output,tgt_mask)


        output = self.fc_layer(result)
        output = output.squeeze(2)
        
        return output

