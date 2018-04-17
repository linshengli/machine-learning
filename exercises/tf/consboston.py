import numpy as np
from sklearn import datasets
import csv

boston = datasets.load_boston()
arr = []
for i in range(boston.data.shape[0]):
    temp = []
    temp = boston.data[i, :]
    temp = np.append(temp, boston.target[i])
    arr.append(temp)
out = open("/home/tbxsx/Code/learnMachineLearning/exercises/tf/boston.csv","w")
mywriter = csv.writer(out)
column = ["feature" + str(i) for i in range(13)]
column.append("Y")
mywriter.writerow(column)
for d in arr:
    mywriter.writerow(d)
out.close()