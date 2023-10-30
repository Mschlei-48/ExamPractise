import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.metrics import mean_squared_error,mean_absolute_error

data=pd.read_csv("tourism.csv")
print(data.head())
#Find missing values
print("Missing values:")
print(data.isnull().sum())
print("Duplicated Rows:")
print(data.duplicated().sum())
#Drop missing values
data=data.dropna()
#Drop duplicated rows
data.drop_duplicates(inplace=True)
#Get info of the data
#print(data.info())
data=pd.get_dummies(data)
#print(data.head())
#print(data.columns)
features=data.iloc[:,1:]
features=features.values
target=data.iloc[:,0]
target=target.values
#print(features.info())
#print(target.info())
x_train,x_test,y_train,y_test=train_test_split(features, target,test_size=0.3)
print(x_train.shape,x_test.shape)

model=RandomForestRegressor()
model.fit(x_train,y_train)
predictions=model.predict(x_test)
mse=mean_squared_error(y_test,predictions)
mae=mean_absolute_error(y_test,predictions)


with open("results.txt","w") as f:
    f.write(f"Mean-Squared Error: {mse} \nMean Absolute Error: {mae}")
with open("Model.pkl","wb") as f:
    pickle.dump(model,f)


