import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest,f_regression
import csv

# load data
data = pd.read_csv("rtt_instance.csv")
X = data.iloc[:, 0:10]
Y_median = data.loc[:, "RTT_median"]
Y_median = Y_median.values.reshape(-1, 1)

Y_average = data.loc[:, "RTT_average"]
Y_average = Y_average.values.reshape(-1, 1)

# 标准化 X,Y
scalerX = preprocessing.StandardScaler().fit(X=X)
X = scalerX.transform(X=X)

scalerY = preprocessing.StandardScaler().fit(X=Y_average)
Y = scalerY.transform(X=Y_average)
Y = Y.ravel()

select = SelectKBest(score_func=f_regression)
X_new = select.fit_transform(X, Y)

print(select.scores_)
print(X_new.shape)

out = open("out2_3.csv","w")
writer = csv.writer(out)
for i in range(10):
    temp = []
    temp.append(data.columns[i])
    temp.append(select.scores_[i])
    writer.writerow(temp)

out.close()

# 使用单个自变量与因变量之间的关系
#[3.80125981e+03 1 1.05080770e+01 7 4.86482423e+00 9 7.95193592e+00 8
# 3.40491927e+03 2 2.10463450e+02 6 2.24895891e+00 10 2.24897856e+02 5
# 5.44082525e+02 3 4.40206826e+02 4]
# netType_num speed destIp_longitude destIp_latitude appname_num synHour carrier_num latitude longitude signal