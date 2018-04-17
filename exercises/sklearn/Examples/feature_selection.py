from sklearn.feature_selection import VarianceThreshold,SelectKBest,f_classif
from sklearn import datasets

X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
sel = VarianceThreshold(0.8 * (1 - 0.8))
sel_X = sel.fit_transform(X)
print(sel_X)
iris = datasets.load_iris()
X,Y = iris.data,iris.target
Ksel = SelectKBest(f_classif, k=3).fit_transform(X, Y)
print(Ksel.shape)
# print(Ksel)

