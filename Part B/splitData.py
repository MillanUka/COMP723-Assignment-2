from sklearn.model_selection import train_test_split
import pandas as pd

# Read the data into a pandas dataframe
data = pd.read_csv('pima-indians-diabetes.data.csv',delimiter=',')   

#Remove the target variables for x. Y only has the target variables
x = data.drop("Class variable (0 or 1)", axis=1)
y = data["Class variable (0 or 1)"]
#Split it. Training has 70% of the records. Test has 30%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

#Combine the target and features together
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

#save the two sets to csv files
train.to_csv("train.csv")
test.to_csv("test.csv")