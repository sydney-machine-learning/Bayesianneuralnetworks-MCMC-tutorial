import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def main():
    raw_data = pd.read_csv("abalone.data")
    raw_data['Sex']= raw_data["Sex"].astype('float64')
    raw_data['Rings']= raw_data["Rings"].astype('float64')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = pd.DataFrame(scaler.fit_transform(raw_data), columns=raw_data.columns)
    train_data, test_data = train_test_split(scaled_data, test_size=0.4, random_state=123)
    train_data.to_csv("train.txt", index=False,header=False, sep=' ')
    test_data.to_csv("test.txt", index=False,header=False, sep=' ')

if __name__ == "__main__":
    main()
