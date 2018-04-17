import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

FEATURE_NUM = 8
Y_LABEL_NAME = "RTT_label"

# netType_num	carrier_num	  longitude     latitude	          speed
# signal	    synHour       appname_num	destIp_longitude      destIp_latitude
# RTT_sum	    RTT_median    RTT_average	RTT_weight_stretch	  RTT_label

'''
D是Y的以为列向量,D为离散值
类型为numpy.ndarray
求解D的信息熵.
'''
def Ent(D, types=None):
    shape = D.shape
    if shape.__len__() != 1:
        print("参数需为列向量")
        return
    total = D.size
    dictMap = {}
    for item in D:
        if dictMap.__contains__(item):
            dictMap[item] += 1
        else:
            dictMap[item] = 1
    valueArr = []
    for (key, value) in dictMap.items():
        pk = (value * 1.0) / total
        valueArr.append(pk)
    return -np.sum(valueArr * np.log2(valueArr))
'''
@:param A是pd的DateFrame,包含所有属性（包括最后的Y） 保证A多于两列
@:param V_name是用来划分的属性名称,是一个属性列表。
@:param C_name是Y值名称
'''

def Gain(A,V_name,C_name):
    totalNum = A.shape[0]
    Y = np.array(A[C_name])
    totalEnt = Ent(Y)
    temp = A.groupby(V_name)
    valueArr = []
    pks = []
    for key,group in temp:
        pks.append((1.0 * group.shape[0]) / totalNum)
        tp = np.array(group[C_name])
        valueArr.append(Ent(tp))
    return totalEnt - np.sum(np.array(pks) * np.array(valueArr))

def Gain1(A,V_name,C_name):
    totalNum = A.shape[0]
    temp = A.groupby(V_name)
    valueArr = []
    pks = []
    for key,group in temp:
        pks.append((1.0 * group.shape[0]) / totalNum)
        tp = np.array(group[C_name])
        valueArr.append(Ent(tp))
    return np.sum(np.array(pks) * np.array(valueArr))


def getImportance():
    data = pd.read_csv("/home/tbxsx/Code/learnMachineLearning/exercises/sklearn/feature_select/stretch_instance_lte_usa_0325.csv")
    featureNum = FEATURE_NUM
    iter = 0
    selectFeatures = []
    #gains是信息增益
    gains = []
    #上一轮的熵
    lastGain = 0
    featureNames = data.columns[0:featureNum].tolist().copy()
    while iter < featureNum:
        if iter == 0:
            #上一层的熵，在根节点处，也就是第0层，熵就等于Y的熵
            lastGain = Ent(np.array(data["RTT_label"]))
        #当前层的熵
        tempthisGain = 1.0*(2**31)
        maxItem = ""
        for item in featureNames:
            #用来分组的数组
            temp = selectFeatures.copy()
            temp.append(item)
            tempGain = Gain1(data, temp, "RTT_label")
            #如果有feature加入当前层的所有feature中后，信息增益更大，则添加该feature。
            if (lastGain - tempGain) > lastGain - tempthisGain:
                maxItem = item
                tempthisGain = tempGain
        iter += 1
        if tempthisGain < 0:
            break
        print("maxItem")
        print(maxItem)
        featureNames.remove(maxItem)
        gains.append(lastGain - tempthisGain)
        #更新上一轮的熵
        lastGain = tempthisGain
        selectFeatures.append(maxItem)
        print("Iter :{0}  Gains".format(iter))
        print(gains)
    return selectFeatures,gains

#[0.2319896568128872, 0.21571043393207634, 0.1960472484984593, 0.08368166160926577, 0.036753805285911795, 0.02638832023327109, 0.018579105811691324, 0.00550871520823724, 0.0, 0.0]
#['destIp_longitude', 'appname_num', 'synHour', 'carrier_num', 'signal', 'destIp_latitude', 'longitude', 'latitude', 'netType_num', 'speed']


features,gains = getImportance()

plt.plot(gains)
plt.show()
print(features)
