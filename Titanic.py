import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn import svm
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
   
    
common = ['Master', 'Mr', 'Miss', 'Mrs']
for df in [train_data, test_data]:
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = [x if x in common else 'Rare' for x in df['Title']]
    
#Fill missing values of age based on mean values of different grouping of PClass+Sex+Title
[df['Age'].fillna(df.groupby(['Pclass','Sex','Title'])['Age'].transform('mean'), inplace=True) for df in [train_data, test_data]]

#Fill missing values of fare based on mean values of different grouping of PClass+embarked
[df['Fare'].fillna(df.groupby(['Pclass','Embarked'])['Fare'].transform('mean'), inplace=True) for df in [train_data, test_data]]

#Remove the two missing values in Embarked
train_data.isnull().sum()
train_data.dropna(subset=['Embarked'], inplace=True)

#Removing PassengerId, Name, Ticket and Cabin, These are removed due to lack of correlation between survival. 
features = ["Pclass", "Sex","Age", "SibSp", "Parch","Fare","Embarked"]
X = train_data[features].copy()
y = train_data["Survived"]
final_test = test_data[features].copy()
final_test = final_test.rename_axis(None, axis=1)

#Changing categorical data to numerical data to allow the performance of numerical operations
y = pd.to_numeric(y, downcast="float")
X['Embarked'].replace('C', 0,inplace=True)
X['Embarked'].replace('Q', 1,inplace=True)
X['Embarked'].replace('S', 2,inplace=True)
final_test['Embarked'].replace('C', 0,inplace=True)
final_test['Embarked'].replace('Q', 1,inplace=True)
final_test['Embarked'].replace('S', 2,inplace=True)
X['Sex'].replace('male', 0,inplace=True)
X['Sex'].replace('female', 1,inplace=True)
final_test['Sex'].replace('male', 0,inplace=True)
final_test['Sex'].replace('female', 1,inplace=True)


#Normalising the data
min_max_scaler = preprocessing.MinMaxScaler()
dummy1 = X.to_numpy() #returns a numpy array
scaled = min_max_scaler.fit_transform(dummy1)
X = pd.DataFrame(scaled)

dummy2 = final_test.to_numpy() #returns a numpy array
scaled2 = min_max_scaler.fit_transform(dummy2)
final_test = pd.DataFrame(scaled2)

#Split the dataset for validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)



def find_Best_Classifier(X_train, X_test, y_train, y_test):
    rand=77
    names = ['RandomForest','MLP','GradientBossting','SVM']
    classifiers = [
                   Pipeline([('RandomForest', RandomForestClassifier(random_state=rand))
                             ]),
                   Pipeline([('MLP', MLPClassifier( random_state=rand,max_iter=2000))
                             ]),
                   Pipeline([('GradientBossting', GradientBoostingClassifier(random_state=rand))
                             ]),
                   Pipeline([('SVM',svm.SVC(kernel='rbf',random_state=rand))
                             ])
                   ]
    i=0
    for name, clf in zip(names, classifiers):
        
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(names[i],round(score*100,2))                 
        i+=1
    return 

#Comparing RF, MLP,GB and kernel-SVM and MLP gives the best result
find_Best_Classifier(X_train, X_test, y_train, y_test)

#Using GridSearch to find the appropriate hyperparameters
MLP =  MLPClassifier(random_state=77,max_iter=2000)
param_grid = { 
    'hidden_layer_sizes': [(10,),(20,),(50,), (100,), (200,),(100,100)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.00005,0.0001, 0.0005],
    'learning_rate': ['constant','adaptive'],
}
clf = GridSearchCV(MLP, param_grid, n_jobs=-1, cv=5, scoring = 'accuracy')
clf.fit(X, y)
print('Best hyperparameters:',clf.best_params_)



#prediction and output as a submission
predictions = clf.predict(final_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
