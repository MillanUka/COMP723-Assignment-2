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
X_train = train.drop("Class variable (0 or 1)", axis=1)
y_train = train["Class variable (0 or 1)"]
X_test = test.drop("Class variable (0 or 1)", axis=1)
y_test = test["Class variable (0 or 1)"]

# from sklearn.tree import DecisionTreeClassifier
# for i in range(2, 20, 2):
#     classifier = DecisionTreeClassifier(max_depth=i)
#     classifier.fit(X_train, y_train)
#     y_pred = classifier.predict(X_test)
#     print(str(i) + " " + str(accuracy_score(y_test, y_pred)))

from sklearn.neural_network import MLPClassifier
x = np.arange(0.001, 0.01, 0.001)
print(x)
for i in x:
    classifier = MLPClassifier(learning_rate_init=i)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(str(i) + " " + str(accuracy_score(y_test, y_pred)))


