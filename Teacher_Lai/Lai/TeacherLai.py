import numpy as np
import pandas as pd

Hitters = pd.read_csv(r'Hitters.CSV')

## x = Hitters的0-12列+18列
## append(arr1, arr2)把arr2加到arr1
## arange(13)输出数组[0,1,..,12]
## append结果就是[0,1,..,12,18]
## iloc[a, b]参数a选取行，参数b选取列
x = Hitters.iloc[:, np.append(np.arange(13), 18)]
print("x:")
print("rows length: ", len(x), "\ncolumns length: ",len(x.columns))

## 删除x中包含Na的行
x2 = x.dropna()
print("\nx2:")
print("rows length: ", len(x2), "\ncolumns length: ",len(x2.columns))

## 求x2每一列的平均数
col_mean = x2.mean()
print("\ncol_mean:\n", col_mean)

## 求平均值中位数, axis = 0求列中位数, axis = 1求行
mean_median = np.median(col_mean, axis = 0)
print("\nmean_median:\n", mean_median)


## 取x2的前13列
x3 = x2.iloc[:, np.append(np.arange(12), 12)]
x3_mean = x3.mean()

## 取x3每一列的的标准差
x3_std = x3.std()
## 标准化
std_x3 = (x3 - x3_mean) / x3_std
## 计算方差
variance = std_x3.var()
print("\nvariance:\n", variance)
## 均值
print("\nstd_x3_mean:\n", std_x3.mean(axis = 0))
print(std_x3.describe().iloc[1, :])
## 结果每一列的均值都是10的-17或-16或-18次方，近似等于0
## 这里有点奇怪，用mean函数求出的平均值和decribe里面的平均值不同, 手算是mean的结果

## 因为std_x3是dataframe类型，不能进行转置，所以先转换为矩阵
A = np.matrix(std_x3)
## 转置
A_T = A.T
B = A_T * A
sum_trace = np.trace(B)
print("\ntrace:\n", sum_trace)
## Salary中心化
cur_y = x2['Salary'] - x2.mean()['Salary']
y = np.array(cur_y)
B_inv = np.linalg.inv(B)
BA = B_inv * A_T
## 一维数组用numpy的转置函数不管用，结果和原来是一样的，需要通过指定shape的方式进行转置为263行1列的y
y.shape = (263, 1)
BAy = BA * y
max_abs = np.abs(BAy).max()
print("\nmax_abs:\n", max_abs)

## 合并x3和y并计算各个列之间的相关关系, 因为之前把x3转成矩阵A丢失了列名，通过columns=重新设置列名
x3_y = pd.DataFrame(np.c_[A, y], columns=np.append(x3.columns.values, 'y'))
relation_x3_y = x3_y.corr()
for i in range(len(relation_x3_y)):
    relation_x3_y.iloc[i, i] = 0
print("\ncorrelation coefficient:\n", relation_x3_y)
## x3中与y相关系数最大的列
print("\nmost relevant to y:\n", relation_x3_y.iloc[:, 13].idxmax())
## x3中相关系数最大的两列
print("\nThe two columns with the largest correlation coefficients:\n", relation_x3_y.stack().idxmax())

D = A[:, 0:2]
DTD = D.T * D
print(DTD)
print(np.linalg.det(DTD))


