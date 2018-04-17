from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn import svm
import numpy as np

# load dataset
# spllit train and test set.
iris = load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=False)

print(X_train.shape)
print(X_test.shape)

# create model
clf = svm.SVC(kernel='linear', C=1.4)

# fit the train data
clf.fit(X_train, Y_train)

# predict
pre = clf.predict(X_test)
print("Predict:")
print(pre)

# get the score
sco = clf.score(X_test, Y_test)
print("score:")
print(sco)

# use cross validation
cv_score = cross_val_score(clf, iris.data, iris.target, cv=20)
print("cv_score")
# print(cv_score)
print(cv_score.mean())
cv_pre = cross_val_predict(clf, iris.data, iris.target,cv=20)
print(cv_pre.shape)
param = {"kernel":('linear', 'rbf'), 'C':np.linspace(0.1,10,100)}

# gridsearchcv
clf = GridSearchCV(clf, param)
clf.fit(iris.data,iris.target)
# print(clf.cv_results_)
print(clf.best_params_)
print(clf.best_score_)
