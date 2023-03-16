import numpy as np
from scipy import signal
from scipy.fftpack import fft, ifft
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# Savitzky-Golay平滑滤波
def SG(data, w, p):
    data = signal.savgol_filter(data, w, p)
    return data

# 傅里叶变换
def FFT(data):
    data = fft(data)
    data = ifft(data)
    return data

# 正态化
def SS(data):
    return StandardScaler().fit_transform(data)

# 均值中心化
def CT(data):
    for i in range(data.shape[0]):
        MEAN = np.mean(data[i])
        data[i] = data[i] - MEAN
    return data

# 多元散射校正
def MSC(data):
    n, p = data.shape
    msc = np.ones((n, p))
    for j in range(n):
        mean = np.mean(data, axis=0)
        # 线性拟合
        for i in range(n):
            y = data[i, :]
            l = LinearRegression()
            l.fit(mean.reshape(-1, 1), y.reshape(-1, 1))
            k = l.coef_
            b = l.intercept_
            msc[i, :] = (y - b) / k
    return msc

# 标准正态变换
def SNV(data):
    m = data.shape[0]
    n = data.shape[1]
    # 求标准差
    data_std = np.std(data, axis=1)  # 每条光谱的标准差
    # 求平均值
    data_average = np.mean(data, axis=1)  # 每条光谱的平均值
    # SNV计算
    data_snv = np.array([[((data[i][j] - data_average[i]) / data_std[i]) for j in range(n)] for i in range(m)])
    return data_snv

# 一阶导数
def D1(data):
    n, p = data.shape
    Di = np.ones((n, p - 1))
    for i in range(n):
        Di[i] = np.diff(data[i])
    return Di

# 二阶导数
def D2(data):
    n, p = data.shape
    Di = np.ones((n, p - 2))
    for i in range(n):
        Di[i] = np.diff(np.diff(data[i]))
    return Di