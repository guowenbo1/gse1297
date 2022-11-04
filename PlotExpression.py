import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import spider as spider
from scipy import stats

#表达量取均值，anova分析使用原始标准数据
def PlotGenExpres (geneList,savePath):

    geneDataSetPath = '/Users/guowenbo/Desktop/OneDrive/生物信息/BioTech/AD标准化数据/AD病标准化数据.xlsx'
    #geneDataSetPath = '/Users/guowenbo/Desktop/OneDrive/生物信息/BioTech/AD病原始数据 整理.xlsx'
    conData = pd.read_excel(geneDataSetPath,sheet_name='Con')
    incData = pd.read_excel(geneDataSetPath, sheet_name='Inc')
    modData = pd.read_excel(geneDataSetPath, sheet_name='Mod')
    sevData = pd.read_excel(geneDataSetPath, sheet_name='Sev')

    #存储表达量的矩阵 返回值
    list1=[]
    for gene in geneList:

        # 存储gene在不同发病程度的表达情况 临时变量
        list = []
        geneID = spider.MicroName(gene)[0]

        conMean,conStd = ConMeanData(geneID,conData)
        list.append(conMean)
        incMean,incStd = IncMeanData(geneID,incData)
        list.append(incMean)
        modMean,modStd = ModMeanData(geneID,modData)
        list.append(modMean)
        sevMean,sevStd = SevMeanData(geneID,sevData)
        list.append(sevMean)

        list1.append(conMean)
        list1.append(incMean)
        list1.append(modMean)
        list1.append(sevMean)

        #画图
        drawBar(list,gene,savePath,[conStd,incStd,modStd,sevStd])
    expresData = np.array(list1).reshape(len(geneList),4)

    return expresData


def GengIDExpression(geneIDList,savePath):
    geneDataSetPath = '/Users/guowenbo/Desktop/OneDrive/生物信息/BioTech/AD标准化数据/AD病标准化数据.xlsx'
    # geneDataSetPath = '/Users/guowenbo/Desktop/OneDrive/生物信息/BioTech/AD病原始数据 整理.xlsx'
    conData = pd.read_excel(geneDataSetPath,sheet_name='Con')
    incData = pd.read_excel(geneDataSetPath, sheet_name='Inc')
    modData = pd.read_excel(geneDataSetPath, sheet_name='Mod')
    sevData = pd.read_excel(geneDataSetPath, sheet_name='Sev')

    #把基因名转换成xxxx_s_at
    path = r'C:\Users\ASUS\OneDrive\桌面\PPT\生物信息\BioTech\ADgene-revised.xls'
    nameList = pd.read_excel(path)

    #存储表达量的矩阵 返回值
    list1=[]
    for name in geneIDList:
        list = []
        conMean,conStd = ConMeanData(name,conData)
        list.append(conMean)
        incMean,incStd = IncMeanData(name,incData)
        list.append(incMean)
        modMean,modStd = ModMeanData(name,modData)
        list.append(modMean)
        sevMean,sevStd = SevMeanData(name,sevData)
        list.append(sevMean)

        list1.append(conMean)
        list1.append(incMean)
        list1.append(modMean)
        list1.append(sevMean)

        # print('is', name,'expression is',list)  # 返还基因对应基因芯片的名字
        # expresList = np.array(list)
        # expresData = np.append(expresData,expresList)

        #画图
        drawBar(list,name,savePath,[conStd,incStd,modStd,sevStd])
    expresData = np.array(list1).reshape(len(geneIDList),4)

    return expresData

#输入一个基因 返还表达值 类型为ndarray
# def CalExpre(gene):
#
#     geneDataSetPath = '/Users/guowenbo/Desktop/OneDrive/生物信息/BioTech/AD标准化数据/AD病标准化数据.xlsx'
#     path = r'C:\Users\ASUS\OneDrive\桌面\PPT\生物信息\BioTech\ADgene-revised.xls'
#     conData = pd.read_excel(geneDataSetPath,sheet_name='Con')
#     incData = pd.read_excel(geneDataSetPath, sheet_name='Inc')
#     modData = pd.read_excel(geneDataSetPath, sheet_name='Mod')
#     sevData = pd.read_excel(geneDataSetPath, sheet_name='Sev')
#     nameList = pd.read_excel(path)
#     name = nameList.loc[nameList['NAME'].isin([gene])]  # 返还基因所那一行的数据
#     name = name.iloc[:, 1:2].values
#     name = str(name)[3:-3]
#     list1 = []
#     list1.append(ConMeanData(name, conData))
#     list1.append(IncMeanData(name, incData))
#     list1.append(ModMeanData(name, modData))
#     list1.append(SevMeanData(name, sevData))
#     list1 = np.array(list1).reshape(1,-1)
#
#     return list1

def drawBar(list,title,path,stdList):

    label =['CON','INC','MOD','SEV']
    title = title.split('_')[0]

    x = np.arange(4)
    # plt.bar(x,list,color=['red','green','orange','blue'],tick_label = label,yerr=stdList)
    plt.bar(x, list, color=['red', 'green', 'orange', 'blue'], tick_label=label)
    plt.title(title)
    fileName = path + '/' + title
    plt.title('HSC70')
    plt.savefig(fileName,dpi=600) #存储文件
    # plt.show()
    plt.close()
    return 0

def CalSTD(con,inc,mod,sev):
    #ddof=0 general std, ddof=1 sample std
    conStd = np.std(con,ddof=0)
    incStd = np.std(inc,ddof=0)
    modStd = np.std(mod,ddof=0)
    sevStd = np.std(sev,ddof=0)
    return conStd,incStd,modStd,sevStd

#返回con正常人的数据and std
def ConMeanData(geneID, conData):

    conData = conData.loc[conData['ID_REF'].isin([geneID])]
    print(geneID)
    conData = conData.iloc[:,1:].values

    #ddof=0 general std, ddof=1 sample std
    std = np.std(conData,ddof=0)
    conData = float(conData.mean(axis=1))


    return conData,std


#返回inc轻度病人的数据
def IncMeanData(geneID, incData):

    incData = incData.loc[incData['ID_REF'].isin([geneID])]
    incData = incData.iloc[:,2:].values
    #ddof=0 general std, ddof=1 sample std
    std = np.std(incData,ddof=0)
    incData = float(incData.mean(axis=1))

    return incData,std


#返回mod中度病人的数据
def ModMeanData(geneID, modData):

    modData = modData.loc[modData['ID_REF'].isin([geneID])]
    modData = modData.iloc[:,2:].values
    #ddof=0 general std, ddof=1 sample std
    std = np.std(modData,ddof=0)
    modData = float(modData.mean(axis=1))

    return modData,std


#返还Sev重度病人的数据
def SevMeanData(geneID, sevData):

    sevData = sevData.loc[sevData['ID_REF'].isin([geneID])]
    sevData = sevData.iloc[:,2:].values
    #ddof=0 general std, ddof=1 sample std
    std = np.std(sevData,ddof=0)
    sevData = float(sevData.mean(axis=1))

    return sevData,std


if __name__ == '__main__':

    allGene = ['BAG2_1','STUB1_1','HSPA8_2','MAPT_3']
    hspa =  ['HSPA12A_1','HSPA12A_2','HSPA14_1','HSPA1L_1','HSPA2_1','HSPA4_1','HSPA4_2','HSPA4_3','HSPA5_1','HSPA6_1','HSPA6_2','HSPA9_1','HSPA9_2','STIP1_1']
    ubiquitin = ['TRAF2_1', 'TRAF5_1', 'TRAF6_1', 'SIKE1_1', 'SIKE1_2', 'SIKE1_3', 'TRAPPC9_1', 'TRAPPC9_2','TRAPPC9_3']
    TAU = ['MAPT_1', 'EML1_1', 'MACF1_1', 'TTBK2_1', 'MAPT_2', 'MAPT_3', 'MAPT_4', 'MAP2_1', 'MAP4_1', 'MARK1_1','MAST1_1']

    hsp72 = ['HSPA1A_1','HSPA1A_2','HSPA1A_3']
    LAMP2 = ['LAMP2_1','LAMP2_2','LAMP2_3']
    proteasome = ['PSMC4_1','PSMC6_1','PSMD1_1','PSMD11_1','PSMD11_2','PSMD12_1','PSMD12_2''PSMD13_1','PSMD13_2','PSMD14_1']
    P62 = ['SQSTM1_1','SQSTM1_2']
    bag = ['BAG1_1','BAG1_2','BAG2_1','BAG3_1','BAG4_1','BAG5_1','BAG5_2']
    LAMP2 = ['LAMP2_1','LAMP2_2']
    APOE = ['APOE_1','APOE_2','APOE_3','APOE_4','APOE_5']
    ATP = ['ATP5F1A_1', 'ATP5F1B_1', 'ATP5F1C_1', 'ATP5F1C_2', 'ATP5F1C_3', 'ATP5F1C_4', 'ATP5F1D_1', 'ATP5F1D_2', 'ATP5F1E_1']
    rab = ['RAB7_1','RAB14_2','RAB14_3','RAB3A_1','RAB3B_1','RAB3B_2']
    chip = ['STUB1_1']
    CASP = ['CASP3_1','CASP6_1','CASP7_1']
    atp1a1 = ['PSMC1_1','PSMC6_1']
    atpase1 = ['ATP1A1_1','ATP1A2_1','ATP1A2_2','ATP1A3_1','ATP1B1_1','ATP1B1_2','ATP1B2_1','ATP1B3_1','ATP1B4_1']
    atp6 = ['ATP6AP1_1''ATP6AP2_1','ATP6AP2_2','ATP6AP2_3']
    all = ['BAG2_1','HSPA8_2','STUB1_1','MAPT_3']
    hsf1 =['HSF1_1','HSF1_2']
    hsc70 = ['HSPA8_2']
    #存储图片
    expresData = PlotGenExpres(hsc70,'/Users/guowenbo/Desktop/OneDrive/生物信息/BioTech/图片/TEST')


    # result = np.corrcoef(expresData)
    # print(type(expresData))
    # print('length',expresData.shape)
    # print(type(result))
    # print(result.shape)
    #
    # corrFileName = '/Users/guowenbo/Desktop/OneDrive/生物信息/BioTech/corrcoef.xlsx'
    # writer = pd.ExcelWriter(corrFileName)
    # result = pd.DataFrame(result)
    # result.to_excel(excel_writer=writer, sheet_name='相关系数')
    # writer.save()


    #计算bag2 与其他基因的cos角度
    # bag2 = CalExpre('BAG2')
    # for i in test:
    #     for j in test:
    #         listi = []
    #         listj = []
    #         listi.append(i)
    #         listj.append(j)
    #         iValue = PlotGenExpres(listi,'/Users/guowenbo/Desktop/OneDrive/生物信息/BioTech/图片/')
    #         jValue = PlotGenExpres(listj,'/Users/guowenbo/Desktop/OneDrive/生物信息/BioTech/图片/')
    #         result = np.corrcoef(iValue,jValue)
    #         print('result',result)
    # BAG2 = CalExpre
    # print(ba)
    # cos = cosine_similarity('BAG2_1','STUB1_1')
    # for i in range(len(allGene)):
    #     gene = allGene[i]
    #     cosValue = cos[0][i]
    #     print('the cos value of BAG2 with',gene,'is',cosValue)





