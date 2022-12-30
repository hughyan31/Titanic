import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
#Loading the data into panda dataframe
train_data = pd.read_csv('train.csv')

test_data = pd.read_csv('test.csv')

#Extracting title from the Name column to give better prediction of missing values of  age
train_data['Title'], test_data['Title'] = [df.Name.str.extract \
        (' ([A-Za-z]+)\.', expand=False) for df in [train_data, test_data]]
    
#Fill missing values of age based on mean values of different combinations of PClass+Sex+Title
[df['Age'].fillna(df.groupby(['Pclass','Sex','Title'])['Age'].transform('mean'), inplace=True) for df in [train_data, test_data]]

#Remove the two missing values in Embarked
train_data.isnull().sum()
[df.dropna(subset=['Embarked'], inplace=True) for df in [train_data, test_data]]

#Removing PassengerId, Name, Ticket and Cabin, These are removed due to lack of correlation between survival. 
features = ["Pclass", "Sex","Age", "SibSp", "Parch","Fare","Embarked"]
X = train_data[features].copy()
y = train_data["Survived"]
X_test = test_data[features].copy()

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


#Normalising the data
z = X.to_numpy() #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(z)
X = pd.DataFrame(x_scaled)


#Filling Missing data based on KNN
imputer = KNNImputer(n_neighbors=2, weights="uniform")
X = imputer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)





tuned_parameters = [
    {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
    {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
]

grid_search = GridSearchCV(
    SVC(), tuned_parameters,
)
grid_search.fit(X_train, y_train)    
    
def Classifier(X_train, X_test, y_train, y_test):
    names = ['RandomForest','MLP','GradientBossting','SVM']
    classifiers = [
                   Pipeline([('RandomForest', RandomForestClassifier(n_estimators=10,max_depth=None,min_samples_split=2, random_state=0))
                             ]),
                   Pipeline([('MLP', MLPClassifier(solver='adam', random_state=0,max_iter=2000))
                             ]),
                   Pipeline([('GradientBossting', GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0))
                             ]),
                   Pipeline([('SVM',svm.SVC(kernel='rbf', C=1))
                             ])
                   ]
    i=0
    for name, clf in zip(names, classifiers):
        
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(names[i],score)                 
        i+=1

    return 


Classifier(X_train, X_test, y_train, y_test)

'''
           


clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print(clf.score(X_test, y_test))

clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))


model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
'''