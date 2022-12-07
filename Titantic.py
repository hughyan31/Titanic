import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn import svm



train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


#Removing PassengerId, Name, Ticket and Cabin, These are removed due to lack of correlation between survival. 
features = ["Pclass", "Sex", "SibSp", "Parch","Fare","Embarked"]
X = train_data[features]
y = train_data["Survived"]
X_test = test_data[features]

#Changing categorical data to numerical data to allow the performance of numerical operations
y = pd.to_numeric(y, downcast="float")
X['Embarked'].replace('C', 0,inplace=True)
X['Embarked'].replace('Q', 1,inplace=True)
X['Embarked'].replace('S', 2,inplace=True)
X_test['Embarked'].replace('C', 0,inplace=True)
X_test['Embarked'].replace('Q', 1,inplace=True)
X_test['Embarked'].replace('S', 2,inplace=True)
X['Sex'].replace('male', 0,inplace=True)
X['Sex'].replace('female', 1,inplace=True)
X_test['Sex'].replace('male', 0,inplace=True)
X_test['Sex'].replace('female', 1,inplace=True)
categoritcal_title=['Sex', 'Embarked']

#Normalising the data
z = X.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(z)
X = pd.DataFrame(x_scaled)


#Filling Missing data based on KNN
imputer = KNNImputer(n_neighbors=2, weights="uniform")
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)



'''
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
'''