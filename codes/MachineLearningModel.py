import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split

def load_data(url):
    try:
        df = pd.read_csv('./data/titanic.csv')
    except Exception:
        df = pd.read_csv(url)
    return df

df_titanic = load_data('https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv')
df_clean_titanic = df_titanic.drop(columns=['PassengerId','Name','Ticket','Cabin'])
df_clean_titanic['Sex'] = df_clean_titanic['Sex'].map({'male': 0, 'female': 1})
df_clean_titanic['Embarked'] = df_clean_titanic['Embarked'].map({'C': 0,'Q': 1,'S': 2})

scaler = MinMaxScaler()
df_clean_titanic['Pclass'] = scaler.fit_transform(df_clean_titanic[['Pclass']])
df_clean_titanic['Age'] = scaler.fit_transform(df_clean_titanic[['Age']])
df_clean_titanic['SibSp'] = scaler.fit_transform(df_clean_titanic[['SibSp']])
df_clean_titanic['Parch'] = scaler.fit_transform(df_clean_titanic[['Parch']])
df_clean_titanic['Fare'] = scaler.fit_transform(df_clean_titanic[['Fare']])
df_clean_titanic['Embarked'] = scaler.fit_transform(df_clean_titanic[['Embarked']])
df_clean_titanic = df_clean_titanic.dropna()
df_clean_titanic = df_clean_titanic[df_clean_titanic['Age'] < 0.824]
df_clean_titanic = df_clean_titanic[df_clean_titanic['SibSp'] < 0.375]
df_clean_titanic = df_clean_titanic[df_clean_titanic['Parch'] < 0.5]
df_clean_titanic = df_clean_titanic[df_clean_titanic['Fare'] < 0.138]

x = df_clean_titanic.drop(columns=['Survived'])
y = df_clean_titanic['Survived']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=99)

knn = KNeighborsClassifier(n_neighbors = 4)
knn.fit(x_train,y_train)

svm_model = SVC(kernel='linear',C=0.1,probability=True)
svm_model.fit(x_train,y_train)

tree_model = DecisionTreeClassifier(random_state=99)
tree_model.fit(x_train,y_train)

voting_model = VotingClassifier(estimators=[
            ('decision_tree', tree_model),
            ('svm', svm_model),
            ('knn', knn)
            ], voting='soft')
voting_model.fit(x_train, y_train)

df_titanic = df_titanic.drop(columns=['PassengerId','Name','Ticket','Cabin'])
df_titanic['Sex'] = df_titanic['Sex'].map({'male': 0, 'female': 1})
df_titanic['Embarked'] = df_titanic['Embarked'].map({'C': 0,'Q': 1,'S': 2})
df_titanic = df_titanic[df_titanic['Age'] < 66]
df_titanic = df_titanic[df_titanic['SibSp'] < 3]
df_titanic = df_titanic[df_titanic['Parch'] < 3]
df_titanic = df_titanic[df_titanic['Fare'] < 71]

scalers = {
    'Pclass': MinMaxScaler().fit(df_titanic[['Pclass']]),
    'Sex': MinMaxScaler().fit(df_titanic[['Sex']]),
    'Age': MinMaxScaler().fit(df_titanic[['Age']]),
    'SibSp': MinMaxScaler().fit(df_titanic[['SibSp']]),
    'Parch': MinMaxScaler().fit(df_titanic[['Parch']]),
    'Fare': MinMaxScaler().fit(df_titanic[['Fare']]),
    'Embarked': MinMaxScaler().fit(df_titanic[['Embarked']]),
}
model_dict = {'model': voting_model, 'scalers': scalers}

with open('supervised.pkl', 'wb') as file:
    pickle.dump(model_dict, file)