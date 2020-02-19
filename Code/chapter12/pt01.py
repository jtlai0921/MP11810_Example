# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.autograd as autograd
import pandas as pd
from torch.autograd import Variable

#用pandas庫中的讀取excel函數讀取上證指數資料
df = pd.read_excel(r"./data/上證指數.xlsx")
df1=df.iloc[:100,3:6].values	# 100x3
xtrain_features=torch.FloatTensor(df1)

df2=df.iloc[1:101,7].values		# 100x1
xtrain_labels=torch.FloatTensor(df2)

xtrain=torch.unsqueeze(xtrain_features,dim=1)
ytrain=torch.unsqueeze(xtrain_labels,dim=1)

x, y = torch.autograd.Variable(xtrain), Variable(ytrain)
class Net(torch.nn.Module):  # 繼承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        # 繼承 __init__ 功能
        # 定義每層用什麼樣的形式
        super(Net, self).__init__()    
        # 隱藏層線性輸出
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   
        self.predict = torch.nn.Linear(n_hidden, n_output)  

    def forward(self, x):  
        # 這同時也是 Module 中的 forward 功能
        # 正向傳播輸入值, 神經網路分析出輸出值
        x = F.relu(self.hidden(x))      
        # 激勵函數(隱藏層的線性值)
        x = self.predict(x)            
        # 輸出值
        return x

# model = Net(n_feature=4, n_hidden=10, n_output=1)
model = Net(n_feature=3, n_hidden=100, n_output=1)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

num_epochs = 1000000
for epoch in range(num_epochs):
    inputs = x
    target = y.unsqueeze(2)
    out = model(inputs)
    # 前向傳播
    loss = criterion(out, target)
    # 計算loss
    optimizer.zero_grad()
    # 梯度歸零
    loss.backward() 
    # 反向傳播
    optimizer.step() 
    # 更新參數
    if (epoch+1) % 20 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'.format(epoch+1,num_epochs,loss.data[0]))

model.eval()
predict = model(x)
predict = predict.data.numpy()
print(predict)
