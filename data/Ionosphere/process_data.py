import pandas as pd
# import sklearn train_test_split
from sklearn.model_selection import train_test_split

def main():
    # raw_train = pd.read_csv("ftrain.csv",header=None)
    # raw_test = pd.read_csv("ftest.csv",header=None)
    # raw_train.to_csv("train.txt", index=False,header=False, sep=' ')
    # raw_test.to_csv("test.txt", index=False,header=False, sep=' ')


    # JS edit 17/03/2023
    raw_data = pd.read_csv("data/Ionosphere/ionosphere.data",header=None)
    # convert the last column to 0 and 1 from text
    raw_data.iloc[:,-1] = raw_data.iloc[:,-1].map({'g':1,'b':0})
    # import matplotlib.pyplot as plt
    # plt.plot(raw_data.iloc[:,-1])
    # plt.savefig("data/Ionosphere/label.png")
    # plt.show()
    raw_train, raw_test = train_test_split(raw_data, test_size=0.3, shuffle=True, random_state=2023)
    raw_train.to_csv("data/Ionosphere/train.txt", index=False,header=False, sep=' ')
    raw_test.to_csv("data/Ionosphere/test.txt", index=False,header=False, sep=' ')


if __name__ == "__main__":
    main()
