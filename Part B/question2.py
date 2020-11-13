from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#Read in the the datasets
# train = pd.read_csv('train.csv',delimiter=',')   
# test = pd.read_csv('test.csv',delimiter=',')   

# from sklearn import preprocessing
# min_max_scaler = preprocessing.MinMaxScaler()
# train = pd.DataFrame(min_max_scaler.fit_transform(train), columns=train.columns)
# test = pd.DataFrame(min_max_scaler.fit_transform(test), columns=test.columns)

#The class variable is 
# X_train = train.drop("Class variable (0 or 1)", axis=1)
# y_train = train["Class variable (0 or 1)"]
# X_test = test.drop("Class variable (0 or 1)", axis=1)
# y_test = test["Class variable (0 or 1)"]


from sklearn.neural_network import MLPClassifier
# classifier = MLPClassifier(hidden_layer_sizes=(20), max_iter=150)
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# print(str(accuracy_score(y_test, y_pred)))

# for i in range(1, 20, 1):
#     classifier = MLPClassifier(hidden_layer_sizes=(20-i, i), max_iter=150)
#     classifier.fit(X_train, y_train)
#     y_pred = classifier.predict(X_test)
#     print("(" + str(20-i)  + ", " + str(i) + ") " + str(accuracy_score(y_test, y_pred)))

data = pd.read_csv('car.data.csv',delimiter=',')   

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in data.columns:
    data[i]=le.fit_transform(data[i])

print(data)
x = data.drop("class", axis=1)
y = data["class"]
#Split it. Training has 70% of the records. Test has 30%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# initialize the baseline classifer 
classifier = MLPClassifier(hidden_layer_sizes=(20), max_iter=150)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(str(accuracy_score(y_test, y_pred)))

for i in range(1, 20, 1):
    classifier = MLPClassifier(hidden_layer_sizes=(20-i, i), max_iter=150)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("(" + str(20-i)  + ", " + str(i) + ") " + str(accuracy_score(y_test, y_pred)))

