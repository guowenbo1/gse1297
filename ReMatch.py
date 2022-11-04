import pandas as pd
import requests
from bs4 import BeautifulSoup

#匹配包含tau蛋白基因 allgenlist：包含基因的列表 savepath：存储的路径
def MatchTau(allGeneList,savePath):

    dataSet = pd.read_excel(allGeneList)

    # regular expression \s空白符
    reTau = '([\s]+tau[\s]+)|(^tau[\s]+)|([\s]+tau[\s]*$)'
    # tau = 'tau\s|\stau$' #17
    result = dataSet[dataSet['Gene Name'].str.contains(reTau,False)]

    geneName = []
    for i in result['Gene Name']:
        a = GeneName(i)
        geneName.append(a)
    result.insert(1,'Name',geneName)
    result.to_excel(savePath)


def GeneName(microName):

    string = microName

    param = {
        'term':string,
    }

    url = requests.get(url='https://www.ncbi.nlm.nih.gov/geoprofiles/', params=param)

    demo = url.text

    soup = BeautifulSoup(demo, 'lxml')  # 用xml方式解码

    result = soup.select('.rsltcont .title')  # results-amount 类别下的value类别的值即 查询的数量

    for i in result:
        i = i.get_text().split('-')
        return i[0]

# def Plot(martixList):
#
#
#     return 0


if __name__ == '__main__':

    allGenPath = 'Gene-name.xls'
    savePath = 'Rematch TAU.xlsx'
    MatchTau(allGenPath,savePath)
