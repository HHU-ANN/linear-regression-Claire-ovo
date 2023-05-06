# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os
try:
    import numpy as np
    from sklearn.linear_model import Ridge, Lasso
except ImportError as e:
    os.system("sudo pip3 install numpy scikit-learn")
    import numpy as np
    from sklearn.linear_model import Ridge, Lasso
def ridge(data):
    x, y = read_data()
    clf = Ridge(alpha=1.0, fit_intercept=True)
    clf.fit(x, y)
    return clf.predict(data.reshape(1, -1))[0]
def lasso(data):
    x, y = read_data()
    clf = Lasso(alpha=0.1, fit_intercept=True, max_iter=10000)
    clf.fit(x, y)
    return clf.predict(data.reshape(1, -1))[0]
def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
