import requests
from bs4 import BeautifulSoup
import xlwt
import pandas as pd


def ResultNFKB(gene):
    string = gene + ' ' + 'NF-'

    param = {
        'term':string,
    }
    url = requests.get(url='https://pubmed.ncbi.nlm.nih.gov/',params=param)

    demo = url.text

    soup = BeautifulSoup(demo,'lxml')

    result = soup.select('.results-amount .value')

    if result:
        for i in result:
            return int(i.get_text())

    #只有一个结果的搜索返回界面不一样
    elif soup.select('.single-result-redirect-message'):
        return 1
    else:
        return 0



def ResultTAU(gene):
    string = gene + ' ' + 'TAU'

    param = {
        'term':string,
    }
    url = requests.get(url='https://pubmed.ncbi.nlm.nih.gov/',params=param)

    demo = url.text

    soup = BeautifulSoup(demo,'lxml')

    result = soup.select('.results-amount .value')

    if result:
        for i in result:
            return int(i.get_text())

    # 只有一个结果的搜索返回界面不一样
    elif soup.select('.single-result-redirect-message'):
        return 1
    else:
        return 0


def ResultUBI(gene):
    string = gene + ' ' + 'UBIQUITIN'

    param = {
        'term':string,
    }
    url = requests.get(url='https://pubmed.ncbi.nlm.nih.gov/',params=param)

    demo = url.text

    soup = BeautifulSoup(demo,'lxml')

    result = soup.select('.results-amount .value')

    if result:
        for i in result:
            return int(i.get_text())

    # 只有一个结果的搜索返回界面不一样
    elif soup.select('.single-result-redirect-message'):
        return 1
    else:
        return 0

def ResultBAG2(gene):
    string = gene + ' ' + 'BAG2'

    param = {
        'term':string,
    }
    url = requests.get(url='https://pubmed.ncbi.nlm.nih.gov/',params=param)

    demo = url.text

    soup = BeautifulSoup(demo,'lxml')

    result = soup.select('.results-amount .value')

    if result:
        for i in result:
            return int(i.get_text())

    # 只有一个结果的搜索返回界面不一样
    elif soup.select('.single-result-redirect-message'):
        return 1
    else:
        return 0
#输入一个芯片名，返回该基因名
def GeneName(microName):

    string = microName

    param = {
        'term':string,
    }

    url = requests.get(url='https://www.ncbi.nlm.nih.gov/geoprofiles/', params=param)

    demo = url.text

    soup = BeautifulSoup(demo, 'lxml')

    result = soup.select('.rsltcont .title')

    for i in result:
        i = i.get_text().split('-')
        return i[0]


#基因名转换为微阵列芯片名 geneList为包含基因名的列表
def MicroName(geneList):
    path = '/Users/guowenbo/Desktop/OneDrive/生物信息/BioTech/hgu133a.xlsx'
    data = pd.read_excel(path,sheet_name='Sheet1')
    list1 = []

    #判断输入数据是否为list
    if isinstance(geneList,list):
        for i in geneList:
            name = data.loc[data['SYMBOL'].isin([i])]
            name = name.iloc[:, 0:1].values
            for name in name:
                name = str(name)
                name = name.replace("'", '')
                name = name.replace("[", '')
                name = name.replace("]", '')
                list1.append(name)
    else:
        name = data.loc[data['SYMBOL'].isin([geneList])]
        name = name.iloc[:, 0:1].values

        for name in name:
            name = str(name)
            name = name.replace("'", '')
            name = name.replace("[", '')
            name = name.replace("]", '')
            list1.append(name)

    return list1

if __name__ == '__main__':

    geneList = ['SMC4','CFHR1','CFH','ZDHHC4','TMA16','TAF1A','FUBP1','USP27X','TMPO']

    nfkbDic = {}
    tauDic = {}
    ubiquitinDic = {}
    bag2Dic = {}
    # HSP70 = ['HSPA6_1']
    # microName = MicroName(HSP70)
    # print(microName)



    for i in geneList:
        nfkbDic[i] = ResultNFKB(i)
        tauDic[i] = ResultTAU(i)
        ubiquitinDic[i] = ResultUBI(i)
        bag2Dic[i] = ResultBAG2(i)

    #写入xlsx
    pathwayList=['ID_REF','NF-KB','TAU','ubiquitin','BAG2']
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('sheet1')

    #写入基因名
    b = 1
    for i in geneList:
        worksheet.write(b, 0, i)
        b = b + 1

    #写入标题栏
    a = 1
    for i in pathwayList:
        worksheet.write(0, a, i)
        a = a + 1

    #写入数据
    for j in range(len(geneList)):
        name = geneList[j]
        j = j + 1
        # worksheet.write(j,1,microName[j-1]) #写入ID_REF名
        worksheet.write(j,2,nfkbDic[name])
        worksheet.write(j, 3, tauDic[name])
        worksheet.write(j, 4, ubiquitinDic[name])
        worksheet.write(j, 5, bag2Dic[name])

    workbook.save('ADgene.xls')  # 保存文件

    print('nfkb is',nfkbDic)
    print('TAU IS',tauDic)
    print('ubiquitin IS', ubiquitinDic)

    # retau = ['203928_x_at','203929_s_at','203930_s_at','206401_s_at','212901_s_at','212905_at','213922_at','216821_at']
    # list = []
    # for i in retau:
    #    list.append(GeneName(i).strip())
    # print(list)

