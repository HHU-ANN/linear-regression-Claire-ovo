# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os
try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np
# 岭回归
def ridge(x):
    X_train, y_train = read_data()
    # 添加一个偏置项
    X_train = np.insert(X_train, 0, values=np.ones(X_train.shape[0]), axis=1)
    x = np.insert(x, 0, values=1)
    # 求解系数
    w = np.linalg.inv(X_train.T.dot(X_train) + np.identity(X_train.shape[1])).dot(X_train.T).dot(y_train)
    # 返回预测值
    return x.dot(w)
# Lasso回归
def lasso(x):
    X_train, y_train = read_data()
    # 添加一个偏置项
    X_train = np.insert(X_train, 0, values=np.ones(X_train.shape[0]), axis=1)
    x = np.insert(x, 0, values=1)
    # 设置超参数和学习率
    alpha = 0.1
    learning_rate = 0.001
    # 初始化权重
    w = np.zeros(X_train.shape[1])
    # 迭代1000次
    for i in range(1000):
        # 计算梯度
        grad = 2 * X_train.T.dot(X_train.dot(w) - y_train) + alpha * np.sign(w)
        # 更新权重
        w = w - learning_rate * grad
    # 返回预测值
    return x.dot(w)
# 读取数据
def read_data(path='./data/exp02/'):
    X_train = np.load(path + 'X_train.npy')
    y_train = np.load(path + 'y_train.npy')
    return X_train, y_train
