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

print(df)

