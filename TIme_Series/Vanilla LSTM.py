#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class lstm(nn.Module):
    def __init__(self):
        super(lstm,self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.tanh = nn.Tanh()
        
        # Input Variable = 7 , hidden_dim = 64 , bidirectional
        self.vlstm = nn.LSTM(7,64,batch_first=True,bidirectional=True,num_layers=3,dropout=0.2)

        self.linear1 = nn.Sequential(
            nn.Linear(128,64),
            nn.Tanh(),
            nn.Linear(64,32),
            nn.Tanh(),
            nn.Linear(32,1)
            )
    
        self.linear2 = nn.Linear(96,24)


    def forward(self,x):
        batch = x.shape[0]
        sequence = x.shape[1]
        variable = x.shape[2]
        out_seq = 24

        x = x.view(batch,sequence,variable)

        out,hidden = self.vlstm(x)

        out = self.linear1(out)
        out = out.squeeze(2)
        out = self.linear2(out)
        
        return out

