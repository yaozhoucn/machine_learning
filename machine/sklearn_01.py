
#线性回归梯度下降法

from sklearn.linear_model import LinearRegression
import numpy as np

# 图形界面库
import matplotlib.pyplot as plt


lr = LinearRegression()


# 损失函数是系数的函数，另外还要传入数据的x，y
def compute_cost(w, b, points):
    total_cost = 0
    M = len(points)
 # 逐点计算平方损失误差，然后求平均数
    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        total_cost += ( y - w * x - b ) ** 2

    return total_cost/M

# 1.引入数据

points = np.genfromtxt('data.csv',delimiter=',')

# 2.提取points中的两组数据，分别作为x，y

# 取所有的行的第一列
x = points[:,0]
# 取所有的行的第二列
y = points[:,1]

x_new = x.reshape(-1, 1)
y_new = y.reshape(-1, 1)

# 3. 定义模型的超参数
alpha = 0.0001
initial_w = 0
initial_b = 0
num_iter = 10

# 4. 定义核心梯度下降算法函数

def grad_desc(points, initial_w, initial_b, alpha, num_iter):
    w = initial_w
    b = initial_b
# 定义一个list保存所有的损失函数值，用来显示下降的过程\n",
    cost_list = []

    for i in range(num_iter):
        cost_list.append( compute_cost(w, b, points) )
        w, b = step_grad_desc( w, b, alpha, points )
        return [w, b, cost_list]

def step_grad_desc( current_w, current_b, alpha, points ):
    sum_grad_w = 0
    sum_grad_b = 0
    M = len(points)
     # 对每个点，代入公式求和\
    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        sum_grad_w += (current_w * x + current_b - y) * x
        sum_grad_b += current_w * x + current_b - y

      # 用公式求当前梯度
    grad_w = 2/M * sum_grad_w
    grad_b = 2/M * sum_grad_b

      # 梯度下降，更新当前的w和b\n",
    updated_w = current_w - alpha * grad_w
    updated_b = current_b - alpha * grad_b

    return updated_w, updated_b

# 5. 测试：运行梯度下降算法计算最优的w和b"

#使用算法库对模型进行训练
lr.fit(x_new, y_new)

#从训练好的模型中提取系数和截距
#w = lr.coef_
#b = lr.intercept_
w1 = lr.coef_[0][0]
b1 = lr.intercept_[0]
cost = compute_cost(w1, b1, points)

# print("w is", w)
# print("b is", b)
# print("cost is", cost)
plt.scatter(x,y)
pred_y = w1 * x + b1
plt.plot(x, pred_y, c='r')
plt.show()