# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os
try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np
def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
def ridge(data):
    # 加载数据
    X_train, y_train = read_data()
    # 添加常数列
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    # 岭回归
    def ridge_regression(X, y, alpha):
        beta = np.linalg.inv(X.T @ X + alpha * np.identity(X.shape[1])) @ X.T @ y
        return beta
    alpha = 1e-10
    beta = ridge_regression(X_train, y_train, alpha)
    # 预测
    data = np.hstack(([1], data))
    data = data.reshape(1, -1)
    prediction = data @ beta  
    return prediction
def lasso(data):
    learning_rate = 0.0000000008
    max_iter = 100000
    alpha = 12000
    X, y = read_data()
    weight = data
    for i in range(max_iter):
        gradient = np.dot(X.T, (np.dot(X, weight) - y)) + alpha * np.sign(weight)
        weight =weight - learning_rate * gradient
    prediction = weight @ data
    return prediction
