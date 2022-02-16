# k近邻算法

#导入算法库
import pandas as pd
import numpy as np


#这里直接映入sklearn里面的数据集，iris
from sklearn.datasets import load_iris

#切分数据集为测试集和训练集
from sklearn.model_selection import train_test_split

#用来计算分类的准确率
from sklearn.metrics import accuracy_score

# 1.数据的加载和预处理

iris = load_iris()

df = pd.DataFrame(data=iris.data,columns=iris.feature_names)

df['class'] = iris.target

df['class'] = df['class'].map({0: iris.target_names[0],1: iris.target_names[1],2: iris.target_names[2]})

#print(df)

#print(df.describe())

x = iris.data
#shape 排成任一行，一列的矩阵
y = iris.target.reshape(-1, 1)



#划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=35,stratify=y)

print(x_train)

#2.核心算法实现

#距离函数的定义
def l1_distance(a,b):
    #把求和的结果保存成一列
    return np.sum(abs(a - b), axis=1)

def l2_distance(a,b):
    return np.sqrt(np.sum( (a-b) ** 2,axis=1))


#核心的分类器实现

class kNN(object):
    #定义一个初始化方法,__init__类的构造方法
    def __init__(self, ):