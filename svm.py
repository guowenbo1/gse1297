import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import utils
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import RocCurveDisplay

np.set_printoptions(suppress=True)


def plot_SVC_decision_function(model, ax=None, plot_support=True):
    '''Plot the decision function for a 2D SVC'''
    if ax is None:
        ax = plt.gca()  # get子图
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    # 生成网格点和坐标矩阵
    Y, X = np.meshgrid(y, x)
    # 堆叠数组
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    print(xy)
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1],
               alpha=0.5, linestyles=['--', '-', '--'])  # 生成等高线 --

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def train_SVM(X,Y,label,model):

    plt.scatter(X,Y,c=label, s=50, cmap='autumn')
    plot_SVC_decision_function(model)
    plt.show()



def plot_3D(X1,X2,y, elev=30, azim=30):
    # 我们加入了新的维度 r
    r = np.exp(-(X1 ** 2).sum(1))
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X1, X2, r, c=y, s=50, cmap='autumn')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def ReadData(filePath,sheetname):

    data = pd.read_excel(filePath,sheet_name=sheetname,index_col=0,header=0)
    data = data.astype('float')
    data = data.T
    geneName = data.columns[0:-1]
    label = data.iloc[:,-1]
    geneData = data.iloc[:,0:-1]
    geneData = preprocessing.scale(geneData,axis=0) #axis=1 363 row mean is 0;
    # scaler = preprocessing.MinMaxScaler()
    # geneData = scaler.fit_transform(geneData)
    geneData = pd.DataFrame(geneData,columns=['BAG2','HSPA8','MAPT','STUB1'])
    return geneName,geneData,label

if __name__ == "__main__":
    dataPath = 'BAG2.xlsx'
    geneName, geneData, label = ReadData(dataPath,sheetname='GSE15222')
    geneName1297,geneData1297,label1297 = ReadData(dataPath,sheetname='GSE1297')
    #geneDataTrain,geneDataTest,labelTrain,labelTest = train_test_split(
    #    geneData,label,shuffle=True,train_size=0.8)
    #print('geneName is',geneName)

    print('geneData is',geneData)


    # forest = RandomForestClassifier()
    # forest = forest.fit(geneData, label)
    # score = forest.score(geneData1297, label1297)
    # print('score is',score)
    #svm
    clf = make_pipeline(StandardScaler(),SVC(gamma='auto'))
    train_sum = geneData.apply(np.sum,axis=1)
    train_std = geneData.apply(np.std,axis=1)
    train_data = pd.DataFrame([train_sum,train_std],index=['sum','std'])
    train_data = train_data.T
    print(train_data)
    clf = clf.fit(train_data,label)
    test_sum = geneData1297.apply(np.sum,axis=1)
    test_std = geneData1297.apply(np.std,axis=1)
    test_data = pd.DataFrame([test_sum,test_std],index=['sum','std'])
    test_data = test_data.T
    print(clf.score(test_data,label1297))

    disp = DecisionBoundaryDisplay.from_estimator(clf, test_data, response_method = "predict",xlabel='Standard Deviation',ylabel='Sum',alpha=1)
    # scatter = disp.ax_.scatter(test_data.iloc[:,0], test_data.iloc[:, 1], c=label1297, edgecolor="k")
    # plt.title('Hyperplane of Support Vector Machines')
    # disp.ax_.legend(*scatter.legend_elements())
    # plt.savefig('svm1.png',dpi=600,format='png')
    # plt.show()
    # plt.close()

    #plot roc
    fig,ax = plt.subplots(1,1)
    svc_disp = RocCurveDisplay.from_estimator(clf, test_data, label1297,name='ROC Curve',color='red',ax=ax)
    ax.plot((0,1),(0,1),transform=ax.transAxes,ls='--',c='k')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.title('Receiver Operating Characteristic Curve')
    plt.savefig('svm2.png',dpi=600)
    plt.show()
