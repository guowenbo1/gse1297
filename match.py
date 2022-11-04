import pandas as pd
import spider as spider

def MatchTau(allGeneList,savePath):

    dataSet = pd.read_excel(allGeneList)

    # regular expression \s空白符
    reTau = '([\s]+tau[\s]+)|(^tau[\s]+)|([\s]+tau[\s]*$)'
    # tau = 'tau\s|\stau$' #17
    result = dataSet[dataSet['Gene Name'].str.contains(reTau,False)]

    geneName = []
    for i in result['Gene Name']:
        a = spider.GeneName(i)
        geneName.append(a)
    result.insert(1,'Name',geneName)
    result.to_excel(savePath)

    return 0

def MatchHsp70(allGeneList,savePath):

    dataSet = pd.read_excel(allGeneList)

    # regular expression \s空白符
    reTau = 'Hsp70|\sHsp70$' #17

    result = dataSet[dataSet['Gene Name'].str.contains(reTau,False)]

    geneName = []
    for i in result['Matrix Name']:
        a = spider.GeneName(i)
        geneName.append(a)
    result.insert(1,'Name',geneName)
    result.to_excel(savePath)

    return 0

if __name__ == '__main__':

    GenePath = r'C:\Users\ASUS\OneDrive\桌面\PPT\生物信息\BioTech\Gene-name.xls'
    MatchHsp70(GenePath,'HSP70.xls')
