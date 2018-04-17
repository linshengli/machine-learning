import pandas as pd
from sklearn import preprocessing
from sklearn import tree
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

clf = tree.DecisionTreeRegressor()
clf.fit(X,Y)
print(clf.feature_importances_)
importances = clf.feature_importances_
out = open("out1_3.csv","w")
writer = csv.writer(out)
for i in range(10):
    temp = []
    temp.append(data.columns[i])
    temp.append(importances[i])
    writer.writerow(temp)

out.close()

#
#[0.0466021   0.22011955 0.02445179 0.03501109 0.02982478
# 0.05841363  0.27095285 0.18077065 0.08141989 0.05243366]

#



