import pandas as pd

def main():
    raw_train = pd.read_csv("ftrain.csv",header=None)
    raw_test = pd.read_csv("ftest.csv",header=None)
    raw_train.to_csv("train.txt", index=False,header=False, sep=' ')
    raw_test.to_csv("test.txt", index=False,header=False, sep=' ')

if __name__ == "__main__":
    main()
