import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#removing empty rows
def clean(df):
    nan_value = float("NaN")
    df.replace("", nan_value, inplace=True)
    df.dropna(subset = ["y"], inplace=True)
    
def draw(df,str):
    plt.figure(figsize = (10,10))
    plt.title(str)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(df['x'],df['y'])
    plt.show()
        
train = pd.read_csv ('/Users/pilot/Desktop/archive/train.csv')
clean(train)
draw(train,"Train Data")

test = pd.read_csv('/Users/pilot/Desktop/archive/test.csv')
clean(test)
draw(test,"Test Data")

#make 2D for scikit-learn.LinearRegression
X_train = train[['x']]
X_test = test[['x']]

Y_train = train['y']
Y_test = test['y']

lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)
accuracy = lin_reg.score(X_test,Y_test)

print(accuracy)

