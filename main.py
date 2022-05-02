import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from os import chdir
chdir("/Users/pilot/Desktop/archive")

# removing empty rows
def clean(df):
    nan_value = float("NaN")
    df.replace("", nan_value, inplace=True)
    df.dropna(subset=["y"], inplace=True)


def draw(df, string):
    plt.figure(figsize=(10, 10))
    plt.title(string, fontsize=16)
    plt.xlabel("x", fontsize=16)
    plt.ylabel("y", fontsize=16)
    plt.scatter(df["x"], df["y"])
    plt.show()


# Training Data
train = pd.read_csv("train.csv")
clean(train)
# draw(train)
X_train = train["x"].values.reshape(-1, 1)
Y_train = train["y"].values.reshape(-1, 1)

# Testing Data
test = pd.read_csv("test.csv")
clean(test)
# draw(test)
X_test = test["x"].values.reshape(-1, 1)
Y_test = test["y"].values.reshape(-1, 1)

lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)

# Plotting using scikit
plt.figure(figsize=(10, 10))
plt.scatter(X_train, Y_train, s=8)
plt.plot(X_train, lin_reg.predict(X_train), color="orange")
plt.title("Train Data", fontsize=20)
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.show()

accuracy = lin_reg.score(X_test, Y_test)
print(accuracy)
