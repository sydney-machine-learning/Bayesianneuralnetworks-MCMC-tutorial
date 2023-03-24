import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    data = pd.read_csv("data/Iris/iris.csv", header=None, delimiter=";")
    # data = np.genfromtxt("iris.csv", delimiter=";")

    scaler = StandardScaler()
    scld = scaler.fit_transform(data)
    # lazy way to keep our outputs nice
    data.iloc[:,:-1] = scld[:,:-1]
    # and we need to go to python indexing
    data.iloc[:,-1] -= 1

    raw_train, raw_test = train_test_split(data, test_size=0.3, shuffle=True, random_state=2023)
    raw_train.to_csv("data/Iris/train.txt", index=False,header=False, sep=' ')
    raw_test.to_csv("data/Iris/test.txt", index=False,header=False, sep=' ')

if __name__ == "__main__":
    main()
