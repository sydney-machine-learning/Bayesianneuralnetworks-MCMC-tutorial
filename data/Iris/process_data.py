import numpy as np
import pandas as pd

def main():
    data = np.genfromtxt("iris.csv", delimiter=";")
    classes = data[:, 4].reshape(data.shape[0], 1) - 1
    features = data[:, 0:4]  # Normalizing Data

    name = "Iris"
    hidden = 12
    input = 4  # input
    output = 3

    for k in range(input):
        mean = np.mean(features[:, k])
        dev = np.std(features[:, k])
        features[:, k] = (features[:, k] - mean) / dev
    train_ratio = 0.7  # choose
    indices = np.random.permutation(features.shape[0])
    traindata = pd.DataFrame(
        np.hstack(
            [
                features[indices[: np.int64(train_ratio * features.shape[0])], :],
                classes[indices[: np.int64(train_ratio * features.shape[0])], :],
            ]
        )
    )
    testdata = pd.DataFrame(
        np.hstack(
            [
                features[indices[np.int64(train_ratio * features.shape[0])] :, :],
                classes[indices[np.int64(train_ratio * features.shape[0])] :, :],
            ]
        )
    )
    traindata.to_csv("train.txt", index=False, header=False, sep=" ")
    testdata.to_csv("test.txt", index=False, header=False, sep=" ")

if __name__ == "__main__":
    main()
