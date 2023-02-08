#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class cnn(nn.Module):
    def __init__(self):
        super(cnn,self).__init__()
        # input variable = 7
        # hidden dim = [3,128]
        self.cnn1 = nn.Sequential(
            nn.Conv1d(7,32,3,2,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=2,padding=1)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(32,128,3,1,padding=1),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.2)

        self.fn = nn.Linear(64,1)
        self.tanh = nn.Tanh()
        self.lstm = nn.GRU(128,64,batch_first=True)
        
        
    def forward(self,x):
        batch = x.shape[0]
        sequence = x.shape[1]
        variable = x.shape[2]
        out_seq = 24
        x = x.view(batch,variable,sequence)
        #CNN INPUT = [BATCH , VARIABLE , SEQUENCE]
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.dropout(x)

        x = x.transpose(1,2)
        x,_ = self.lstm(x)
        x = self.fn(x)
        x = self.tanh(x)

        x = x.view(batch,out_seq)
        return x

