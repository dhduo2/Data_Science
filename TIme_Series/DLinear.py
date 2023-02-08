#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 이동평균을 구하는 MA class
# AvgPool1d로 일정 구간의 평균을 구하며, 끝값을 동일하게 패딩하는 처리가 들어간다.
class ma(nn.Module):
    def __init__(self,kernel_size):
        super(ma,self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size,stride=1,padding=0)

    def forward(self,x):
        front = x[:,0:1,:].repeat(1,(self.kernel_size-1)//2 , 1)
        end = x[:,-1:,:].repeat(1,(self.kernel_size-1)//2 , 1)
        x = torch.cat((front,x,end),dim=1)

        moving_average = self.avg(x.permute(0,2,1))
        x = moving_average.permute(0,2,1)
        
        return x

# 시계열 데이터를 Trend와 그외 나머지로 구분짓는 Decomposition
class decomposition(nn.Module):
    def __init__(self,kernel_size):
        super(decomposition,self).__init__()
        self.ma = ma(kernel_size)

    def forward(self,x):
        average = self.ma(x)
        remainder = x-average

        return average , remainder


class DLinear(nn.Module):
    def __init__(self,kernel_size,variable,inp_seq,out_seq):
        super(DLinear,self).__init__()
        self.decomp = decomposition(kernel_size)
        
        # trend 와 remainder를 처리할 각각의 Linear layer 필요
        # 각각의 변수에 별로 다른 Layer 처리가 필요하므로 Modulist사용
        
        trend_lst = [nn.Linear(inp_seq, out_seq) for i in range(variable)]
        self.trend = nn.ModuleList(trend_lst)
    
        remainder_lst = [nn.Linear(inp_seq, out_seq) for i in range(variable)]
        self.remainder = nn.ModuleList(trend_lst)

        self.fn_layer = nn.Sequential(
            nn.Tanh(),
            nn.Linear(variable,1)

        )
    def forward(self,x):
    
        batch = x.shape[0]
        sequence = x.shape[1]
        variable = x.shape[2]
        output_sequence = 24

        avg,rem = self.decomp(x)

        dec_avg = torch.zeros(batch,output_sequence,variable).to(device)
        dec_rem = torch.zeros(batch,output_sequence,variable).to(device)
        for i in range(variable):
            inp_avg = avg[:,:,i]
            inp_rem = rem[:,:,i]

            dec_avg[:,:,i] = self.trend[i](inp_avg)
            dec_rem[:,:,i] = self.remainder[i](inp_rem)

        fn_output = dec_avg + dec_rem
        fn_output = self.fn_layer(fn_output)
        fn_output = fn_output.squeeze(2)
    
        return fn_output

