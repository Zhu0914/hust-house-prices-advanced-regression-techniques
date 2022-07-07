import numpy as np
import pandas as pd
class LinearRegression:
    def __init__(self):
        '''初始化模型'''
        self.coef_ = None
        self.interception_ = None
        self._theta = None

    def fit(self, X_train, y_train):
        '''根据训练数据集X_train,y_train训练模型'''
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict(self, X_predict):
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def __repr__(self):
        return 'LinearRegression()'

#1、读取数据
train_house= pd.read_csv('train.csv')

#2、数据预处理
#第一列数据是ID，对房价并无影响，先把它单独抽离出来。
train_house.drop("Id", axis=1, inplace=True)
"""
2.1 缺失值处理
分析含缺失值特征的作用，没用的特征直接删除，有用的特征依据缺失量，少则删除样本，多则用mean,median或mod补全；
首先，如果缺失率达到15%以上，那这项特征应该予以删除并认为数据集中不存在这样的特征，既假定它是不存在的，因此，本数据集删除的特征有：’PoolQC’,‘MiscFeature’,‘Alley’，‘Fence’，‘FireplaceQu’和‘LotFrontage’这几列。
其次，在剩下的含缺失值变量中，以Garage开头的5个GarageX特征具有相同数量的缺失值，据此推测他们可能代表的是同一组观测值，而关于Garage的信息，’GarageCars’已经能够很好地表征了，因此删除这几个特征，对BsmtX也可以进行同样的操作。
之后，对于MasVnrArea和MasVnrType，它们与YearBuilt和OverallQual有较强的相关性。因此，删除这两个特征也不会丢失任何信息。
然后，除了Electrical，其它无意义的含缺失值的变量都已经删除了，Electrical这个变量下只有一个样本带有缺失值，因此不妨删除带有这个缺失值的那各样本。
"""

na_count = train_house.isnull().sum().sort_values(ascending=False)#得到各个特征的缺失量
na_rate = na_count / len(train_house)#计算各个特征的缺失率
na_data = pd.concat([na_count,na_rate],axis=1,keys=['count','ratio'])


train_house.drop(na_data[na_data['count'] > 1].index, axis=1, inplace=True)
train_house.drop(train_house.loc[train_house['Electrical'].isnull()].index, inplace=True)

"""
2.2 字符串型特征映射为数值型特征
factorize函数可以将Series中的标称型数据映射称为一组数字，相同的标称型映射为相同的数字，将所有的非数值型数据转换为数值型数据：
"""
for col in train_house.columns:
    if train_house[col].dtypes == "object":
        train_house[col], uniques = pd.factorize(train_house[col])

#3、建模

X = train_house.drop('SalePrice', axis=1)#自变量
y = train_house["SalePrice"]#因变量

model=LinearRegression()
model=model.fit(X,y)

test_house = pd.read_csv("test.csv")

#test数据集进行处理
test_house_ID = test_house["Id"]
test_house.drop("Id", axis=1,  inplace=True)
test_house.drop(na_data[na_data['count'] > 1].index, axis=1, inplace=True)
for col in test_house.columns:
    if test_house[col].dtypes == "object":
        test_house[col], uniques = pd.factorize(test_house[col])
    test_house[col].fillna(test_house[col].mean(), inplace=True)

#预测房价
test_predict= model.predict(test_house)
save_result=pd.concat([test_house_ID,pd.Series(abs(test_predict))],axis=1,keys=["Id","SalePrice"])
save_result.to_csv("submisson.csv",index=False)
