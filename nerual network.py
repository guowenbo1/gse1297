import random
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
from sklearn.utils import shuffle
import pickle


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

def AllGeneData(geneName):

    genePath = '/Users/guowenbo/Desktop/OneDrive/生物信息/BioTech/Neural Network Data.xlsx'

    geneID = spider.MicroName(geneName)[0]
    geneData = pd.read_excel(genePath,header=None,index_col=0)
    geneData = geneData.T
    # print('tranpose geneData is',geneData)
    geneData = geneData[['MMSE', geneID]]

    # geneData = geneData.groupby('MMSE', as_index=False).agg('mean')  # 相同的MMSE取均值 as_index=False不把MMSE列设为行索引
    # geneData[geneID] = round(geneData[geneID], 5)  # 保留5位小数点

    return geneData

#read data as pytorch class
def ReadData(filePath):
    Data = pd.read_excel(filePath, sheet_name='ORG',header=None)
    # geneData = Data.iloc[1:,1]
    # geneData= geneData.astype('float')
    # geneData = geneData.to_numpy()
    # geneData = preprocessing.scale(geneData)
    # geneData = torch.from_numpy(geneData)
    MMSE = Data.iloc[1:,5]
    BAG2 = Data.iloc[1:,1]
    CHIP = Data.iloc[1:,2]
    HSPA8 = Data.iloc[1:,3]
    TAU = Data.iloc[1:,4]

    BAG2 = BAG2.astype('float')
    CHIP = CHIP.astype('float')
    HSPA8 = HSPA8.astype('float')
    TAU = TAU.astype('float')
    MMSE = MMSE.astype('float')
    BAG2 = BAG2.to_numpy()
    CHIP = CHIP.to_numpy()
    HSPA8 = HSPA8.to_numpy()
    TAU = TAU.to_numpy()
    MMSE = MMSE.to_numpy()
    BAG2 = shuffle(BAG2)
    CHIP = shuffle(CHIP)
    HSPA8 = shuffle(HSPA8)
    TAU = shuffle(TAU)
    MMSE = shuffle(MMSE)
    BAG2 = preprocessing.scale(BAG2)
    BAG2 = torch.from_numpy(BAG2)

    CHIP = preprocessing.scale(CHIP)
    CHIP = torch.from_numpy(CHIP)

    HSPA8 = preprocessing.scale(HSPA8)
    HSPA8 = torch.from_numpy(HSPA8)

    TAU = preprocessing.scale(TAU)
    TAU = torch.from_numpy(TAU)


    MMSE = torch.from_numpy(MMSE)
    # geneData= geneData.astype('float')
    # geneData = geneData.to_numpy()
    # geneData = preprocessing.scale(geneData)
    # geneData = torch.from_numpy(geneData)

    return BAG2,CHIP,HSPA8,TAU,MMSE
    # return geneData,MMSE


if __name__ == '__main__':
    # trainingFile = '/Users/guowenbo/Desktop/OneDrive/生物信息/BioTech/nnTrainingData.xlsx'
    # testFile = '/Users/guowenbo/Desktop/OneDrive/生物信息/BioTech/nnTestData.xlsx'
    # geneName = ['BAG2_1', 'STUB1_1', 'HSPA8_2', 'MAPT_3']
    # genePath = '/Users/guowenbo/Desktop/OneDrive/生物信息/BioTech/Neural Network Data.xlsx'
    #
    # BAG2,CHIP,HSPA8,TAU,MMSE = ReadData(genePath)
    # BAG2Test =BAG2[25:]
    # CHIPTest = CHIP[25:]
    # HSPA8Test = HSPA8[25:]
    # TauTest = TAU[25:]
    # MMSETest = MMSE[25:]
    # BAG2 = BAG2[:24]
    # CHIP = CHIP[:24]
    # HSPA8 = HSPA8[:24]
    # TAU = TAU[:24]
    # MMSE = MMSE[:24]
    # #
    # x = torch.cat([BAG2,CHIP,HSPA8,TAU])
    # # # x, y = ReadData(trainingFile)
    # x = x.view(-1,4)
    # # # y = torch.cat([TAU,MMSE])
    # y = MMSE
    # y = y.view(-1,1)
    # # print(x.shape,MMSE.shape)
    # myNet = nn.Sequential(
    #         nn.Linear(4,32),
    #         nn.Sigmoid(),
    #         # nn.Linear(32,64),
    #         # nn.Sigmoid(),
    #         # nn.Linear(64, 128),
    #         # nn.Sigmoid(),
    #         # nn.Linear(128, 256),
    #         # nn.Sigmoid(),
    #         nn.Linear(32,1),
    # )
    # myNet = torch.load('myNet(0.95).pkl')
    # optimzer = torch.optim.SGD(myNet.parameters(),lr=0.005)
    # loss_func = nn.MSELoss()
    # deal_dataset = TensorDataset(x,y)
    # tr_set = DataLoader(dataset=deal_dataset,batch_size=3,shuffle=True)
    #
    # for epoch in range(1000):
    #     myNet.train()
    #     for x1,y1 in tr_set:
    #         optimzer.zero_grad()
    #         out = myNet(x1)
    #         loss = loss_func(out,y1)
    #         # print('loss is',loss)
    #         # print('y1',y1)
    #         # print('out is',out)
    #         # print(myNet.state_dict())
    #         loss.backward()
    #         optimzer.step()
    #     if epoch % 100 == 0:
    #         print("Epoch:{},Loss:{:.4f}".format(epoch, loss))
    # # print('result',myNet(x))
    # # print('result',np.corrcoef(myNet(x).detach().numpy()))
    # print('training model corrcoef',np.corrcoef(myNet(x).detach().numpy()[:,0],y[:,0]))
    #
    #
    # # a = np.array([4462.7],dtype='float')
    # # print('x is',x.numpy())
    # # print('x out',myNet(x).detach().numpy())
    # # print('x1 is',x1)
    # # print('x1 out',myNet(x1))
    # # a = torch.from_numpy(a)
    # # a = torch.tensor(a).float()
    # # print('out',myNet(a))
    # # print('MSE',MMSE)
    # # plt.scatter(x=MMSE.numpy(),y=myNet(x).detach().numpy())
    # # plt.show()
    #
    #
    # #evaluation function
    # myNet.eval()
    # total_loss =0
    # # # BAG2Test,CHIPTest,HSPA8Test,TauTest, MMSETest = ReadData(testFile)
    # xTest = torch.cat([BAG2Test,CHIPTest,HSPA8Test,TauTest])
    # # # xTest,yTest = ReadData(testFile)
    # yTest = MMSETest
    # # # yTest = torch.cat([TauTest,MMSETest])
    # xTest = xTest.view(-1,4)
    # yTest = yTest.view(-1,1)
    # with torch.no_grad():
    #     pred = myNet(xTest)
    #     lossValue = loss_func(pred,yTest)
    #     total_loss += lossValue.cpu().item()*len(xTest)
    #     avg_loss = total_loss/len(xTest)
    #     print('pred data', pred)
    # print('total loss is',total_loss)
    # print('avgloss',avg_loss)
    # print('predict data corrcoef', np.corrcoef(pred.detach().numpy()[:, 0],yTest[:,0] ))
    #
    # #plot
    # plt.scatter(range(len(y)+len(yTest)),np.append(MMSE,MMSETest))
    # plt.plot(range(len(y)),myNet(x).data)
    # print(range(len(y),len(y)+len(yTest)))
    # print(myNet(xTest).data)
    # plt.plot(range(len(y),len(y)+len(yTest)),myNet(xTest).data,color='r')
    # plt.tick_params(direction='in',width=2,length=4)
    #
    #
    # #save bp nn
    # torch.save(myNet,'myNet.pkl')
    #
    # writer = pd.ExcelWriter('myNet Data.xlsx')
    # pd.DataFrame(x.detach().numpy()).to_excel(excel_writer=writer, sheet_name='training data')
    # pd.DataFrame(y.detach().numpy()).to_excel(excel_writer=writer, sheet_name='true labels')
    # pd.DataFrame(myNet(x).data.detach().numpy()).to_excel(excel_writer=writer, sheet_name='training labels')
    # pd.DataFrame(xTest.detach().numpy()).to_excel(excel_writer=writer, sheet_name='test data')
    # pd.DataFrame(yTest.detach().numpy()).to_excel(excel_writer=writer,sheet_name='test labels')
    # pd.DataFrame(pred.detach().numpy()).to_excel(excel_writer=writer, sheet_name='predict labels')
    # writer.save()
    # print('write success')
    # writer.close()
    # plt.show()
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

    #load pkl
    myNet = torch.load('myNet(0.95).pkl')
    data_filename = 'myNet Data(0.95).xlsx'
    x = pd.read_excel(data_filename,sheet_name='training data')
    y = pd.read_excel(data_filename,sheet_name='true labels')
    xTest = pd.read_excel(data_filename,sheet_name='test data')
    yTest = pd.read_excel(data_filename,sheet_name='test labels')
    predict_labels = pd.read_excel(data_filename,sheet_name='predict labels')
    traing_labels = pd.read_excel(data_filename,sheet_name='training labels')

    # plt.scatter(range(len(y)+len(yTest)),np.append(y,yTest))
    # plt.plot(range(len(y)),traing_labels,label='Training')
    # plt.xlabel('Patient Number')
    # plt.ylabel('MMSE')
    # plt.plot(range(len(y),len(y)+len(yTest)),predict_labels,color='r',label = 'Test')
    # plt.tick_params(direction='in',width=2,length=4)
    # frame = plt.legend(bbox_to_anchor=(0.79, 0.3)).get_frame()
    # frame.set_linewidth(0)
    # frame.set_facecolor('none')
    # # plt.legend(bbox_to_anchor=(0.78, 0.3)).get_frame().set_linewidth(0)
    # # plt.legend(bbox_to_anchor=(0.8, 0.3))
    # # plt.legend('boxoff') #去掉边框
    # plt.axvline(x=24, ls='--', c='green')
    # plt.savefig('NN1.png',dpi=300)
    # plt.show()
    # plt.close()

    #boxplot
    # print(([yTest.values.tolist()],[predict_labels.values.tolist()]))
    box_data = yTest
    box_data['predict_labels'] = predict_labels
    print(box_data)
    plt.boxplot(box_data,widths=0.5,labels=['MMSE','Predict MMSE'])
    plt.title('Boxplot of MMSE and Predict MMSE')
    # plt.boxplot(yTest,positions=[0],widths = 0.5,patch_artist='True',labels = 'MMSE')
    # plt.boxplot(predict_labels,positions=[1],widths=0.5,patch_artist='True', labels = 'Predict MMSEr')
    plt.savefig('nn2.png',dpi=600)
    # plt.show()