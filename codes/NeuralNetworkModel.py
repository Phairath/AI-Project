import pandas as pd
import keras
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler
import sklearn.model_selection as model_selection

def load_data():
    try:
        df = pd.read_csv('./data/road_accident_survival.csv')
    except Exception:
        print('Error to Open the Dataset')
    return df

df_accident = load_data()
df_accident['Gender'] = df_accident['Gender'].map({'Male': 0,'Female': 1})
df_accident['Seatbelt'] = df_accident['Seatbelt'].map({'No': 0,'Yes': 1})
df_accident['Airbag_Deployment'] = df_accident['Airbag_Deployment'].map({'Not Deployed': 0,'Deployed': 1})
df_accident['Alcohol_Involved'] = df_accident['Alcohol_Involved'].map({'No': 0,'Yes': 1})
df_accident['Weather_Condition'] = df_accident['Weather_Condition'].map({'Clear': 0,'Snow': 1,'Rain': 2,'Fog':3})
scaler = MinMaxScaler()
df_accident['Age'] = scaler.fit_transform(df_accident[['Age']])
df_accident['Speed_of_Impact'] = scaler.fit_transform(df_accident[['Speed_of_Impact']])
df_accident['Weather_Condition'] = scaler.fit_transform(df_accident[['Weather_Condition']])
df_accident = df_accident.dropna()
df_accident = df_accident[df_accident['Age'] < 0.886]
df_accident = df_accident[df_accident['Speed_of_Impact'] < 0.616]

model = keras.Sequential()
model.add(keras.Input(shape=(7,)))
model.add(keras.layers.Dense(32,activation = 'relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(keras.layers.Dense(16,activation = 'relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(keras.layers.Dense(8,activation = 'relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(keras.layers.Dense(1,activation = 'sigmoid'))
# model.summary()
# optimizer = Adam(learning_rate=0.001)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
x = df_accident.drop(columns=['Survived'])
y = df_accident['Survived']
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.80, random_state=99)
history_model = model.fit(x_train,y_train,epochs = 80,validation_data=(x_test, y_test))
model.evaluate(x_test,y_test)
model.save('neural.keras')