from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
#Read in the the datasets
train = pd.read_csv('train.csv',delimiter=',')   
test = pd.read_csv('test.csv',delimiter=',')   

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
train = pd.DataFrame(min_max_scaler.fit_transform(train), columns=train.columns)
test = pd.DataFrame(min_max_scaler.fit_transform(test), columns=test.columns)

#The class variable is 
X_train = train.drop(["Class variable (0 or 1)", "Unnamed: 0"], axis=1)
y_train = train["Class variable (0 or 1)"]
X_test = test.drop(["Class variable (0 or 1)", "Unnamed: 0"], axis=1)
y_test = test["Class variable (0 or 1)"]

from sklearn.tree import DecisionTreeClassifier
# for i in range(2, 20, 2):
#     classifier = DecisionTreeClassifier(max_depth=i)
#     classifier.fit(X_train, y_train)
#     y_pred = classifier.predict(X_test)
#     print(str(i) + " " + str(accuracy_score(y_test, y_pred)))

from sklearn.neural_network import MLPClassifier
# x = np.arange(0.001, 0.01, 0.001)
# for i in x:
#     classifier = MLPClassifier(learning_rate_init=i)
#     classifier.fit(X_train, y_train)
#     y_pred = classifier.predict(X_test)
#     print(str(i) + " " + str(accuracy_score(y_test, y_pred)))

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# for k in range(2, 7, 1):
#     fs = SelectKBest(chi2, k=k)
#     fs.fit(X_train, y_train)
#     X_train_fs = fs.transform(X_train)
#     X_test_fs = fs.transform(X_test)
#     classifier = DecisionTreeClassifier(max_depth=2)
#     classifier.fit(X_train_fs, y_train)
#     y_pred = classifier.predict(X_test_fs)
#     print(str(k) + " " + str(accuracy_score(y_test, y_pred)))

# for k in range(2, 7, 1):
#     fs = SelectKBest(chi2, k=k)
#     fs.fit(X_train, y_train)
#     X_train_fs = fs.transform(X_train)
#     X_test_fs = fs.transform(X_test)
#     classifier = MLPClassifier(learning_rate_init=0.004)
#     classifier.fit(X_train_fs, y_train)
#     y_pred = classifier.predict(X_test_fs)
#     print(str(k) + " " + str(accuracy_score(y_test, y_pred)))    

classifier = DecisionTreeClassifier(max_depth=2)
fs = SelectKBest(chi2, k=4)
fs.fit(X_train, y_train)
X_train_fs = fs.transform(X_train)
X_test_fs = fs.transform(X_test)
classifier.fit(X_train_fs, y_train)

prob = classifier.predict_proba(X_test_fs)



classifier = MLPClassifier(learning_rate_init=0.004)
fs = SelectKBest(chi2, k=6)
fs.fit(X_train, y_train)
X_train_fs = fs.transform(X_train)
X_test_fs = fs.transform(X_test)
classifier.fit(X_train_fs, y_train)

prob = classifier.predict_proba(X_test_fs)
print(prob)
print(prob[0])
print(classifier.classes_)