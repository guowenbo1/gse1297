import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import linear_fitting as lf
import spider as spider
import nft

# 返回基因geneID的基因表达量，按照MMSE分类，且对把相同的MMSE均值处理
def AllGeneData(geneName, genePath):

    geneID = spider.MicroName(geneName)[0]
    geneData = pd.read_excel(genePath,header=None,index_col=0)
    geneData = geneData.T
    print('tranpose geneData is',geneData)
    geneData = geneData[['MMSE', geneID]]

    geneData = geneData.groupby('MMSE', as_index=False).agg('mean')  # 相同的MMSE取均值 as_index=False不把MMSE列设为行索引
    geneData[geneID] = round(geneData[geneID], 5)  # 保留5位小数点

    return geneData, geneID


# 剔除异常值， 异常值为不在mean±2std
def RemAbnormal(geneData,geneID):
    mean = geneData.iloc[:][geneID].mean()
    std = geneData.iloc[:][geneID].std()
    normal = [mean + 2 * std, mean - 2 * std]
    print('normal is', normal)
    # lambda 匿名函数 transofrm和apply相似针对每一列运算
    geneData['two_sigma'] = geneData[geneID].transform(
        lambda x: (x.mean() - 2 * x.std() > x) | (x.mean() + 2 * x.std() < x))
    geneData = geneData[geneData['two_sigma'] == False]
    geneData = geneData.drop(columns=['two_sigma'])

    return geneData


# 返还一个两列的矩阵，第一列为MMSE的差值，第二列为该差值表达量的差异的平均值
def ExpDifOnMMSE(geneData):
    geneID = geneData.columns[1]
    countMMSE = len(geneData)
    difMatrix = pd.DataFrame(columns=['DifMMSE', 'expression'])

    for i in range(countMMSE):  # i is 第i个病人
        orgData = geneData.iloc[i:, :]
        a = geneData.iloc[i]  # I病人的表达量
        a = np.tile(a, (countMMSE - i, 1))  # 平铺a矩阵成countMMSE-I行
        dif = orgData - a
        dif = dif.abs() #abs value
        dif[geneID] = round(dif[geneID], 5)
        dif.rename(columns={'MMSE': 'DifMMSE', geneID: 'expression'}, inplace=True)  # inplace对源数据修改
        difMatrix = pd.concat([difMatrix, dif], axis=0)  # 末尾添加dif矩阵

    difMatrix = difMatrix.groupby('DifMMSE', as_index=False).agg('mean')

    return difMatrix.iloc[1:,:]

def LinearFit(alpha,epslion,x,y):

    theat0 = 0  # 第一个参数
    theat1 = 0  # 第二个参数
    error0 = 0
    count = 0
    error = []

    while True:
        count += 1
        theat0, theat1 = lf.gradient_descent(theat0, theat1, alpha, x, y)
        error1 = lf.MSE(theat0, theat1, x, y)  # 计算迭代的误差
        error.append(error1)
        if abs(error1 - error0) <= epslion:
            break
        else:
            error0 = error1
    return count, theat0,theat1,error


if __name__ == '__main__':

    geneName = ['BAG2_1','STUB1_1','TRAF5_1','HSPA9_1','MAPT_3']
    # geneName = ['BAG2_1']
    genePath = '/Users/guowenbo/Desktop/OneDrive/生物信息/BioTech/MMSE&NFT拟合用数据.xlsx'
    photoPath = '/Users/guowenbo/Desktop/OneDrive/生物信息/BioTech/图片/MMSE'
    for gene in geneName:
        alpha = 0.003  # 学习率
        epslion = 0.00001  # 最小误差值 当两次迭代的MSE小于此值时停止迭代
        expresData,geneID = AllGeneData(gene, genePath)
        print(expresData)
        #原始数据画图
        x =expresData[:]['MMSE']
        y=expresData[geneID]
        plt.scatter(x=expresData[:]['MMSE'],y=expresData[geneID])
        plt.xlabel('MMSE')
        ylabel = gene + ' MMSE Expression'
        plt.ylabel(ylabel)
        plt.title(gene)
        #linear fit
        count,theat0,theat1,error = LinearFit(alpha=alpha,epslion=epslion,x=x,y=y)
        yy = theat0 + theat1*x
        plt.plot(x,yy)
        fileName = photoPath +'/' +ylabel
        plt.savefig(fileName)
        # plt.show()
        plt.close()

        #
        # # 画图：未清理异常数据
        # print('expresData is', expresData)
        # difData = ExpDifOnMMSE(expresData)
        # print('difData is', difData)
        # x = difData[:]['DifMMSE'].values
        # y = difData[:]['expression'].values
        # count,theat0,theat1,error = LinearFit(alpha=alpha,epslion=epslion,x=x,y=y)
        # print('for non 2sigma data:迭代次数',count,'theat0:',theat0,'theat1:',theat1)
        # yy = theat0 + theat1*x
        # plt.scatter(x, y)
        # plt.plot(x,yy)
        # plt.xlabel('Difference in MMSE')
        # plt.ylabel('Difference in Gene Expression')
        # title = gene + ' Expression Base on DifMMSE without 2-sigma'
        # plt.title(title)
        # fileName = photoPath + '/'+ title
        # plt.savefig(fileName)
        # # plt.show()
        # plt.close()
        #
        # # 画图：已清理异常数据
        # cleanData = RemAbnormal(expresData,geneID)
        # print('cleanData is', cleanData)
        # difCleanData = ExpDifOnMMSE(cleanData)
        # print('difCleanData is', difCleanData)
        # x = difCleanData[:]['DifMMSE'].values
        # y = difCleanData[:]['expression'].values
        # count, theat0, theat1, error = LinearFit(alpha=alpha, epslion=epslion, x=x, y=y)
        # print('for 2sigma data:迭代次数', count, 'theat0:', theat0, 'theat1:', theat1)
        # yy = theat0 + theat1 * x
        # plt.scatter(x, y)
        # plt.xlabel('Difference in MMSE')
        # plt.ylabel('Gene Expression Difference')
        # title = gene + ' Gen Expression Base on DifMMSE with 2-sigma'
        # plt.title(title)
        # plt.plot(x, yy, label="my model",color='red')
        # fileName = photoPath + '\\'+title
        # plt.savefig(fileName)
        # # plt.show()
        # plt.close()
        # # 绘制损失函数的变化
        # plt.plot(error)  # x值缺省时默认为len(error)
        # plt.ylabel('Cost J')
        # plt.xlabel('Iterations')
        # plt.title('cost function in 2-sigma')
        # # plt.show()
        # plt.close()
    # #画图MMSE和NFT的
    # MMSEData,ID = AllGeneData(geneName, genePath)
    # NFTData,ID = nft.AllGeneData(geneName,genePath)
    # x=MMSEData[:]['MMSE'].values
    # y=NFTData[:]['NFT'].values
    # count = range(len(x))
    # print('count is',count)
    # print('x is',x)
    # plt.scatter(count,x)
    # plt.xlabel('patient')
    # plt.ylabel('MMSE')
    # plt.title('MMSE')
    # fileName = photoPath + 'MMSE'
    # plt.savefig(fileName)
    # plt.show()
    #
    # plt.scatter(count,y)
    # plt.xlabel('patient')
    # plt.ylabel('NFT')
    # plt.title('NFT')
    # fileName = photoPath + 'NFT'
    # plt.savefig(fileName)
    # plt.show()