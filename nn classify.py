import torch
torch.set_default_tensor_type(torch.DoubleTensor)
import torch.nn as nn
import numpy as np
import pandas as pd
import spider as spider
from torch.utils.data import Dataset, DataLoader,TensorDataset
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


class MyDataset(Dataset):
    def __init__(self):
        file = '/Users/guowenbo/Desktop/OneDrive/生物信息/BioTech/nnTrainingData.xlsx'
        Data = pd.read_excel(file, sheet_name='ORG', header=None)
        geneData = Data.iloc[:, [0, 1, 2, 3, 4]]
        BAG2 = geneData.iloc[1:, 1]
        CHIP = geneData.iloc[1:, 2]
        HSPA8 = geneData.iloc[1:, 3]
        TAU = geneData.iloc[1:, 4]
        MMSE = Data.iloc[1:, 5]
        BAG2 = BAG2.astype('float')
        CHIP = CHIP.astype('float')
        HSPA8 = HSPA8.astype('float')
        TAU = TAU.astype('float')
        MMSE = MMSE.astype('float')
        BAG2 = BAG2.to_numpy()
        BAG2 = torch.from_numpy(BAG2)
        CHIP = CHIP.to_numpy()
        CHIP = torch.from_numpy(CHIP)
        HSPA8 = HSPA8.to_numpy()
        HSPA8 = torch.from_numpy(HSPA8)
        TAU = TAU.to_numpy()
        TAU = torch.from_numpy(TAU)
        MMSE = MMSE.to_numpy()
        MMSE = torch.from_numpy(MMSE)
        x = torch.cat([BAG2, CHIP, HSPA8, TAU])
        # x = x.view(-1,4)
        self.x_data = x
        self.y_data = MMSE

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.len
# #
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.net = nn.Sequential(
            #4个输入
            nn.Linear(4,32),
            nn.Sigmoid(),
            #30个输出
            nn.Linear(32,30)
        )
    def forward(self,x):
        return self.net(x)


#read data as pytorch class
def ReadData(filePath):
    Data = pd.read_excel(filePath, sheet_name='ORG_OLD',header=None)
    # geneData = Data.iloc[1:,1]
    # geneData= geneData.astype('float')
    # geneData = geneData.to_numpy()
    # geneData = preprocessing.scale(geneData)
    # geneData = torch.from_numpy(geneData)
    group = Data.iloc[1:,6:8]
    BAG2 = Data.iloc[1:,1]
    CHIP = Data.iloc[1:,2]
    HSPA8 = Data.iloc[1:,3]
    TAU = Data.iloc[1:,4]
    #
    BAG2 = BAG2.astype('float')
    CHIP = CHIP.astype('float')
    HSPA8 = HSPA8.astype('float')
    TAU = TAU.astype('float')
    group = group.astype('float')
    BAG2 = BAG2.to_numpy()
    BAG2 = preprocessing.scale(BAG2)
    BAG2 = torch.from_numpy(BAG2)
    CHIP = CHIP.to_numpy()
    CHIP = preprocessing.scale(CHIP)
    CHIP = torch.from_numpy(CHIP)
    HSPA8 = HSPA8.to_numpy()
    HSPA8 = preprocessing.scale(HSPA8)
    HSPA8 = torch.from_numpy(HSPA8)
    TAU = TAU.to_numpy()
    TAU = preprocessing.scale(TAU)
    TAU = torch.from_numpy(TAU)

    group = group.to_numpy()
    group = torch.from_numpy(group)
    # geneData= geneData.astype('float')
    # geneData = geneData.to_numpy()
    # geneData = preprocessing.scale(geneData)
    # geneData = torch.from_numpy(geneData)

    return BAG2,CHIP,HSPA8,TAU,group
    # return geneData,group


if __name__ == '__main__':
    trainingFile = '/Users/guowenbo/Desktop/OneDrive/生物信息/BioTech/nnTrainingData.xlsx'
    testFile = '/Users/guowenbo/Desktop/OneDrive/生物信息/BioTech/nnTestData.xlsx'
    geneName = ['BAG2_1', 'STUB1_1', 'HSPA8_2', 'MAPT_3']
    genePath = '/Users/guowenbo/Desktop/OneDrive/生物信息/BioTech/Neural Network Data.xlsx'

    BAG2,CHIP,HSPA8,TAU,MMSE = ReadData(trainingFile)
    x = torch.cat([BAG2,CHIP,HSPA8,TAU])
    y = MMSE
    # x, y = ReadData(trainingFile)
    print('x',x)
    print('y',y)
    x = x.view(-1,4)
    # y = torch.cat([TAU,MMSE])
    y = y.view(-1,2)
    # print(x.shape,MMSE.shape)
    myNet = nn.Sequential(
            nn.Linear(4,32),
            nn.Sigmoid(),
            # nn.Linear(32,64),
            # nn.Sigmoid(),
            # nn.Linear(64, 128),
            # # nn.Sigmoid(),
            # nn.Linear(128, 256),
            # nn.Sigmoid(),
            nn.Linear(32,2),
            nn.Softmax(dim=1),
    )
    optimzer = torch.optim.SGD(myNet.parameters(),lr=0.005)
    loss_func = nn.CrossEntropyLoss()
    # loss_func = nn
    deal_dataset = TensorDataset(x,y)
    tr_set = DataLoader(dataset=deal_dataset,batch_size=3,shuffle=True)
    correct =0
    total =0
    for epoch in range(100):
        myNet.train()
        for x1,y1 in tr_set:
            print('y1',y1)
            optimzer.zero_grad()
            out = myNet(x1)
            print('out is', out.data)
            # print(out.argmax(1))
            loss = loss_func(out,y1)
            print('loss is',loss)
            loss.backward()
            optimzer.step()
            # predicted = torch.max(out.data,1)[1]
            # print('predicted',predicted)
            # total += y1.size(0)
            # correct += (predicted == y1).sum().item()
            # print('correct',correct)
            # print('accuracy',100 * correct / total)

        if epoch % 100 == 0:
            print("Epoch:{},Loss:{:.4f}".format(epoch, loss))
    # print('result',myNet(x))
    # # print('result',np.corrcoef(myNet(x).detach().numpy()))
    # print('corrcoef',np.corrcoef(myNet(x).detach().numpy()[:,0],y[:,0]))


    #evaluation function
    myNet.eval()
    total_loss =0
    BAG2Test,CHIPTest,HSPA8Test,TauTest, MMSETest = ReadData(testFile)
    xTest = torch.cat([BAG2Test,CHIPTest,HSPA8Test,TAU])
    # xTest,yTest = ReadData(testFile)
    # # yTest = MMSETest
    # # yTest = torch.cat([TauTest,MMSETest])
    # xTest = xTest.view(-1,1)
    # yTest = yTest.view(-1,4)
    # # xTest = xTest.squeeze(-1)
    # with torch.no_grad():
    #     pred = myNet(xTest)
    #     lossValue = loss_func(pred,yTest)
    #     total_loss += lossValue.cpu().item()*len(xTest)
    #     avg_loss = total_loss/len(xTest)
    #     print('pred data', pred)
    # print('total loss is',total_loss)
    # print('avgloss',avg_loss)
    # print('corrcoef', np.corrcoef(pred.detach().numpy()[:, 0],yTest[:,0] ))
    # # print(myNet.state_dict())


    #
    # device = torch.device("cpu")
    # dataset = MyDataset()
    # tr_set = DataLoader(dataset,batch_size=2,shuffle=True)
    # print('trset',tr_set)
    # model = MyModel().to(device)
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(),0.1)
    # #
    # for epoch in range(100):
    #     model.train()
    #     for x,y in tr_set:
    #         optimizer.zero_grad()
    #         x,y = x.to(device),y.to(device)
    #         pred = model(x)
    #         loss = criterion(pred,y)
    #         loss.backward()
    #         optimizer.step()