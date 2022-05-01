import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

train = pd.read_csv ('/Users/pilot/Desktop/archive/train.csv')

#removing empty,NaN rows
nan_value = float("NaN")
train.replace("", nan_value, inplace=True)
train.dropna(subset = ["y"], inplace=True)

X_train = train['x'].values.reshape(-1,1)
Y_train = train['y'].values.reshape(-1,1)

plt.figure(figsize = (10,10))
plt.scatter(X_train,Y_train)
plt.show()

test = pd.read_csv('/Users/pilot/Desktop/archive/test.csv')
X_test = test['x'].values.reshape(-1,1)
Y_test = test['x'].values.reshape(-1,1)

lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)
accuracy = lin_reg.score(X_test,Y_test)

print(accuracy)
