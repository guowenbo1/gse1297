import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import spider as spider


def AllGeneData(geneName,conData,incData,modData,sevData):

    geneID= spider.MicroName(geneName)[0]
    print('geneID',geneID)
    conData = ConData(geneID,conData)
    incData = IncData(geneID,incData)
    modData = ModData(geneID,modData)
    sevData = SevData(geneID,sevData)

    # #Normalization
    # conData = UnitNormalization(conData).astype('float')
    # incData = UnitNormalization(incData).astype('float')
    # modData = UnitNormalization(modData).astype('float')
    # sevData = UnitNormalization(sevData).astype('float')

    # #max min normalization
    # conData = MaxMinNormalization(conData).astype('float')
    # incData = MaxMinNormalization(incData).astype('float')
    # modData = MaxMinNormalization(modData).astype('float')
    # sevData = MaxMinNormalization(sevData).astype('float')

    #standardization
    conData = Standardization(conData).astype('float')
    # print('geneName is',geneName,'condata',conData)
    incData = Standardization(incData).astype('float')
    # print('geneName is', geneName, 'incdata', incData)
    modData = Standardization(modData).astype('float')
    # print('geneName is', geneName, 'moddata', modData)
    sevData =Standardization(sevData).astype('float')
    # print('geneName is', geneName, 'sevdata', sevData)

    return conData,incData,modData,sevData


#最大最小归一化
def MaxMinNormalization(data):
    max = np.amax(data)
    min = np.amin(data)
    data = (data - min) / (max - min)
    return data

#百分比化
def PercentageNormalization(data):
    sum = data.sum()
    data = data/sum
    return data


#standardation
def Standardization(data):
    mean = data.mean()
    #ddof=0 general std, ddof=1 sample std
    std = np.std(data,ddof=0)
    data = (data-mean)/std
    return data


#单位化 round in ten decimals
def UnitNormalization(data):
    length = np.linalg.norm(data)
    data = data / length
    data = data.astype(float)
    data = np.round(data,decimals=10)
    return data


#返回con正常人的数据
def ConData(geneID, conData):
    conData = conData.loc[conData['ID_REF'].isin([geneID])]
    conData = conData.iloc[:,1:].values
    return conData


#返回inc轻度病人的数据
def IncData(geneID, incData):

    incData = incData.loc[incData['ID_REF'].isin([geneID])]
    incData = incData.iloc[:,1:].values

    return incData


#返回mod中度病人的数据
def ModData(geneID, modData):

    modData = modData.loc[modData['ID_REF'].isin([geneID])]
    modData = modData.iloc[:,1:].values

    return modData


#返还Sev重度病人的数据
def SevData(geneID, sevData):

    sevData = sevData.loc[sevData['ID_REF'].isin([geneID])]
    sevData = sevData.iloc[:,1:].values

    return sevData


#返还四个矩阵（con,inc,mod,sev)
def InnerMatrix(geneList,conData,incData,modData,sevData):
    geneData = pd.DataFrame(index=['con','inc','mod','sev'],columns=geneList)
    for i in range(len(geneList)):
        conData1,incData1,modData1,sevData1 = AllGeneData(geneList[i],conData,incData,modData,sevData)
        geneData[geneList[i]] =[conData1,incData1,modData1,sevData1]
    conValue = []
    incValue = []
    modValue = []
    sevValue = []
    for i in range(len(geneList)):
        for j in range(len(geneList)):
            #循环求内积
            innerValue = Product(geneData.loc[:,geneList[i]],geneData.loc[:,geneList[j]])
            conValue.append(np.sum(innerValue[0]))
            incValue.append(np.sum(innerValue[1]))
            modValue.append(np.sum(innerValue[2]))
            sevValue.append(np.sum(innerValue[3]))
    conValue =np.array(conValue).reshape(len(geneList),len(geneList))
    conValue = pd.DataFrame(data=conValue,columns=geneList,index=geneList)
    incValue = np.array(incValue).reshape(len(geneList), len(geneList))
    incValue = pd.DataFrame(data=incValue, columns=geneList, index=geneList)
    modValue = np.array(modValue).reshape(len(geneList), len(geneList))
    modValue = pd.DataFrame(data=modValue, columns=geneList, index=geneList)
    sevValue = np.array(sevValue).reshape(len(geneList), len(geneList))
    sevValue = pd.DataFrame(data=sevValue, columns=geneList, index=geneList)

    fileName = ''
    for n in geneList:
        fileName = fileName + '&&'+ n

    dirPath = photoPath + '/' + fileName
    # create dir
    os.chdir(photoPath)
    if not os.path.isdir(dirPath):
        os.mkdir(fileName)

    # print('con innerproduct',conValue)
    # print('inc innerprodcut',incValue)
    # print('mod innerproduct',modValue)
    # print('sev innerproduct',sevValue)

    #write xlsx
    excelName = dirPath + '/' +fileName+ 'InnerProduct.xls'
    writer = pd.ExcelWriter(excelName)
    conValue.to_excel(excel_writer=writer,sheet_name='con')
    incValue.to_excel(excel_writer=writer,sheet_name='inc')
    modValue.to_excel(excel_writer=writer,sheet_name='mod')
    sevValue.to_excel(excel_writer=writer,sheet_name='sev')
    writer.save()
    writer.close()

    conEigvalue,conEigvector = np.linalg.eig(conValue)
    conEigvalue = PercentageNormalization(conEigvalue)
    incEigvalue,incEigvector = np.linalg.eig(incValue)
    incEigvalue = PercentageNormalization(incEigvalue)
    modEigvalue,modEigvector = np.linalg.eig(modValue)
    modEigvalue = PercentageNormalization(modEigvalue)
    sevEigvalue,sevEigvector = np.linalg.eig(sevValue)
    sevEigvalue = PercentageNormalization(sevEigvalue)
    return conEigvalue,incEigvalue,modEigvalue,sevEigvalue


#返回对应元素相乘的值
def Product(a,out=None):
    # 迭代器
    with np.nditer([a,out],
            flags = ['external_loop', 'buffered','refs_ok'],
            op_flags = [['readonly'],['readonly']
                        # ['writeonly', 'allocate', 'no_broadcast']
                        ]) as it:
        z = np.ndarray((4,),dtype=float)
        for x, y in it:
            z = x*y
        return z


def SortValue(geneNameList,allEigvalue,titleName):
    fileName = ''
    sort = np.array(allEigvalue)
    sort = sort.reshape((1,4*len(allEigvalue[0])))
    sort = -np.sort(-sort)
    flag = True if sort[0][0] > 0.7 else False
    for n in geneNameList:
        fileName = fileName + '&&'+ n
    if flag:
        for i in range(len(allEigvalue)):
            for j in range(i, len(allEigvalue)):
                # 特征值排序
                sortValue = np.array([allEigvalue[i], allEigvalue[j]])
                sortValue = sortValue.reshape((1, 2 * len(allEigvalue[0])))
                #descend order
                sortValue = -np.sort(-sortValue)
                # print('the eigvalue of', titleName[i], 'compare with', titleName[j], 'is', sortValue)
                title = 'The Eigvalue of ' + titleName[i].title() + ' Compare With ' + titleName[j].title()
                savePath = photoPath + '/' + fileName +'/'+title
                value1Index=[]
                value2Index=[]
                for k in allEigvalue[i]:
                    iIndex = np.where(sortValue[0] == k)
                    np.ravel(iIndex)
                    value1Index.append(iIndex[0].tolist())
                for m in allEigvalue[j]:
                    JIndex = np.where(sortValue[0] == m)
                    value2Index.append(JIndex[0].tolist())
                DrawBar(allEigvalue[i],value1Index,titleName[i],allEigvalue[j],value2Index,titleName[j],title,savePath,geneNameList)
    else:
        savePath = photoPath + '/' + fileName
        shutil.rmtree(savePath)
    return 0


def DrawBar(value1,value1Index,value1Label,value2,value2Index,value2Label,title,savePath,geneNameList):
    if len(value1Index[0]) !=1:
        for i in range(len(value1Index)):
            plt.bar(value1Index[i],value1[i])
            for j in range(len(value1Index[i])):
                #数字偏右 -0.25抵消偏移
                plt.text(value1Index[i][j]-0.25,value1[i],value1[i].round(4),weight='bold')
    else:
        for a in range(len(value1Index)):
            if a ==0:
                plt.bar(value1Index[a],value1[a],color='r',label=value1Label)
            else:
                plt.bar(value1Index[a], value1[a], color='r')
            for c in range(len(value1Index[a])):
                plt.text(value1Index[a][c]-0.25,value1[a],value1[a].round(4),weight='bold')
        for b in range(len(value2Index)):
            if b== 0:
                plt.bar(value2Index[b],value2[b],color='b',label=value2Label)
            else:
                plt.bar(value2Index[b], value2[b], color='b')
            for d in range(len(value2Index[b])):
                plt.text(value2Index[b][d]-0.25,value2[b],value2[b].round(4),weight='bold')

    plt.ylabel('Eigvalue')
    # plt.xlabel(geneNameList)
    plt.xlabel('BAG2 STUB1 HSC70 MAPT')
    plt.title(title)
    plt.legend()
    plt.savefig(savePath,dpi=600)
    # plt.show()
    plt.close()
    return 0

if __name__ == '__main__':
    dataPath = '/Users/guowenbo/Desktop/OneDrive/生物信息/BioTech/AD标准化数据/AD病标准化数据.xlsx'
    # dataPath = '/Users/guowenbo/Desktop/OneDrive/生物信息/BioTech/AD病原始数据 整理.xlsx'
    photoPath = '/Users/guowenbo/Desktop/OneDrive/生物信息/BioTech/图片/report/innerproduct/600dpi'
    # photoPath = r'C:\Users\ASUS\OneDrive\桌面\PPT\生物信息\BioTech\test'
    conData = pd.read_excel(dataPath, sheet_name='Con')
    incData = pd.read_excel(dataPath, sheet_name='Inc')
    modData = pd.read_excel(dataPath, sheet_name='Mod')
    sevData = pd.read_excel(dataPath, sheet_name='Sev')

    # # #TEST
    # # path = r'C:\Users\ASUS\OneDrive\桌面\PPT\生物信息\BioTech\GPL96基因对照表.xlsx'
    # # data = pd.read_excel(path,sheet_name='Sheet1')
    # # name = ['con', 'inc', 'mod', 'sev']
    # # import random
    # # x = 0
    # # while x<6:
    # #     a = random.randint(1,19000)
    # #     b = random.randint(1,19000)
    # #     c = random.randint(1,19000)
    # #     d =random.randint(1,19000)
    # #     e =random.randint(1,19000)
    # #     f = random.randint(1,19000)
    # #     g = random.randint(1,19000)
    # #     h=random.randint(1,19000)
    # #     i=random.randint(1,19000)
    # #     a = data.iloc[a][1]
    # #     b = data.iloc[b][1]
    # #     c = data.iloc[c][1]
    # #     d = data.iloc[d][1]
    # #     e = data.iloc[e][1]
    # #     f = data.iloc[f][1]
    # #     g = data.iloc[g][1]
    # #     h=data.iloc[h][1]
    # #     i=data.iloc[i][1]
    # #     list = [a,b,c,d]
    # #     conEigvalue, incEigvalue, modEigvalue, sevEigvalue = InnerMatrix(list, conData,incData,modData,sevData,list)
    # #     allEigvalue = [conEigvalue,incEigvalue,modEigvalue,sevEigvalue]
    # #     print('for the gene',list)
    # #     SortValue(list,allEigvalue,name)
    # #     x +=1
    # #     ##TEST END

    #所有蛋白质的折叠
    bag2 = ['BAG2_1']
    ubiquitin =['TRAF2_1','TRAF5_1','TRAF6_1','SIKE1_1','SIKE1_2','SIKE1_3','TRAPPC9_1','TRAPPC9_2','TRAPPC9_3']
    hsp70 = ['HSPA12A_1','HSPA12A_2','HSPA14_1','HSPA1L_1','HSPA2_1','HSPA4_1','HSPA4_2','HSPA4_3','HSPA5_1','HSPA6_1','HSPA6_2','HSPA9_1','HSPA9_2','STIP1_1']
    TAU=['MAPT_1','EML1_1','MACF1_1','TTBK2_1','MAPT_2','MAPT_3','MAPT_4','MAP2_1','MAP4_1','MARK1_1','MAST1_1']
    chipGene = ['STUB1_1']
    # hsc70 = ['HSPA8_1','HSPA8_2','HSPA8_3']
    hsc70 = ['HSPA8_2']
    ATP = ['ATP5F1A_1', 'ATP5F1B_1', 'ATP5F1C_1', 'ATP5F1C_2']
    geneList = ['BAG2_1','STUB1_1','HSPA8_2','MAPT_3']
    name = ['con', 'inc', 'mod', 'sev']

    con=[]
    inc = []
    mod = []
    sev = []

    max = 0
    for bag in bag2:
        for chip in chipGene:
            for hsp in hsc70:
                for tau in ['MAPT_3']:
                            list = [bag,chip,hsp,tau]
                            conEigvalue, incEigvalue, modEigvalue, sevEigvalue = InnerMatrix(list, conData,incData,modData,sevData)
                            allEigvalue = [conEigvalue,incEigvalue,modEigvalue,sevEigvalue]
                            SortValue(list,allEigvalue,name)


