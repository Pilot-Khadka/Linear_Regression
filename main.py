import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#removing empty rows
def clean(df):
    nan_value = float("NaN")
    df.replace("", nan_value, inplace=True)
    df.dropna(subset = ["y"], inplace=True)
        
train = pd.read_csv ('/Users/pilot/Desktop/archive/train.csv')
clean(train)

test = pd.read_csv('/Users/pilot/Desktop/archive/test.csv')
clean(test)

#make 2D
X_train = train[['x']]
X_test = test[['x']]

Y_train = train['y']
Y_test = test['y']

plt.figure(figsize = (10,10))
plt.xlabel("X_train")
plt.ylabel("Y_train")
plt.scatter(X_train,Y_train)
plt.show()

lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)
accuracy = lin_reg.score(X_test,Y_test)

print(accuracy)

