import numpy as np
import pandas as pd
import spider as spider
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Activation
# from tensorflow.keras.optimizers import RMsprop
import tensorflow as tf
# devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(devices[0],True)


#read data as pytorch class
def ReadData(filePath):
    Data = pd.read_excel(filePath, sheet_name='ORG',header=None)
    geneData = Data.iloc[1:,1]
    MMSE = Data.iloc[1:,-1]

    # BAG2 = geneData.iloc[1:,1]
    # CHIP = geneData.iloc[1:,2]
    # HSPA8 = geneData.iloc[1:,3]
    # TAU = geneData.iloc[1:,4]

    # BAG2 = BAG2.astype('float')
    # CHIP = CHIP.astype('float')
    # HSPA8 = HSPA8.astype('float')
    # TAU = TAU.astype('float')
    MMSE = MMSE.astype('float')
    # BAG2 = BAG2.to_numpy()
    # BAG2 = torch.from_numpy(BAG2)
    # CHIP = CHIP.to_numpy()
    # CHIP = torch.from_numpy(CHIP)
    # HSPA8 = HSPA8.to_numpy()
    # HSPA8 = torch.from_numpy(HSPA8)
    # TAU = TAU.to_numpy()
    # TAU = torch.from_numpy(TAU)
    geneData= geneData.astype('float')
    MMSE = MMSE.to_numpy()
    geneData = geneData.to_numpy()
    geneData = geneData.reshape(-1,1)
    geneData = normalize(geneData,axis=0,norm='max')


    # return BAG2,CHIP,HSPA8,TAU,MMSE
    return geneData,MMSE


if __name__ == '__main__':
    trainingFile = r'C:\Users\ASUS\OneDrive\生物信息\BioTech\nnTrainingData.xlsx'
    testFile = r'C:\Users\ASUS\OneDrive\生物信息\BioTech\nnTestData.xlsx'
    geneName = ['BAG2_1', 'STUB1_1', 'HSPA8_2', 'MAPT_3']
    # genePath = '/Users/guowenbo/Desktop/OneDrive/生物信息/BioTech/Neural Network Data.xlsx'
    # x, MMSE = ReadData(trainingFile)
    x = np.linspace(-1,1-200)
    np.random.shuffle(x)
    MMSE = 0.5*x + 2
    print('x is',x)
    print('mmse',MMSE)
    # model =Sequential([
    #     #     Dense(32,input_dim=1),
    #     #     Activation('relu'),
    #     #     Dense(1),
    #     # ])
    model = Sequential()
    model.add(Dense(units=1,input_shape=(1,)))
    model.compile(optimizer='sgd',loss='mse')
    for step in range(100):
        cost = model.train_on_batch(x,MMSE)
        print('cost',cost)