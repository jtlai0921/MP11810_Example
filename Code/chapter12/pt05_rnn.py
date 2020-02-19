#匯入所需的模組
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import pandas as pd

#設定參數
input_size = 1
hidden_size = 100
num_layers = 10
num_classes = 1

df = pd.read_excel(r"./data/上證指數.xlsx")
df1 = df.iloc[:100,3:6].values
xtrain_features=torch.FloatTensor(df1)

df2 = df["漲跌"].astype(float)
xtrain_labels=torch.FloatTensor(df2[:100])

xtrain = torch.unsqueeze(xtrain_features,dim=1)
ytrain = torch.unsqueeze(xtrain_labels,dim=1)

x1 = torch.autograd.Variable(xtrain_features.view(100,3,1))
y = torch.autograd.Variable(ytrain)

#定義遞歸神經網路結構
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

rnn = RNN(input_size, hidden_size, num_layers, num_classes)
#rnn = RNN(100, 1, 10, 1)
use_gpu = torch.cuda.is_available()  # 判斷是否有GPU加速
#if use_gpu:
#    rnn = rnn.cuda()

#損失函數以及最佳化函數
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.005)

#訓練模型
for epoch in range(1000):  #10000
    inputs = x1
    target = y
    out = rnn(inputs) # 前向傳播
    loss = criterion(out, target) # 計算loss
    # backward
    optimizer.zero_grad() # 梯度歸零
    loss.backward() # 方向傳播
    optimizer.step() # 更新參數

    if (epoch+1) % 20 == 0:
        print('Epoch[{}], loss: {:.6f}'.format(epoch+1,loss.data[0]))

rnn.eval()
predict = rnn(x1)
predict = predict.data.numpy()
print(predict)
