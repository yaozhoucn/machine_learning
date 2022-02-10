

# 简单线性回归，最小二乘法

# 0.引入依赖
# 算法库
import numpy as np

# 图形界面库
import matplotlib.pyplot as plt

# 1.引入数据

points = np.genfromtxt('data.csv',delimiter=',')

#print(points[0, 0:2])



# 2.提取points中的两组数据，分别作为x，y

# 取所有的行的第一列
x = points[:,0]
# 取所有的行的第二列
y = points[:,1]

# 用plt画出散点图
plt.scatter(x,y)
plt.show()

# 定义一个损失函数
# 损失函数是一个系数函数，另外还要传入数据的x，y
def compute_cost(w ,b ,points):
    total_cost = 0
    m = len(points)

    # 逐点计算平方损失误差，然后求平均数
    for i in range(m):
        x = points[i,0]
        y = points[i,1]
        total_cost += (y -w * x - b) ** 2
    return total_cost/m

# 定义核心算法，拟合函数
#def average(data):