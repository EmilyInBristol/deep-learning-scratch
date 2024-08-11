import requests
import os
import pandas as pd  # type: ignore
import hashlib

url_train = 'http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_train.csv'
sha1_hash_train = '585e9cc93e70b39160e7921475f9bcd7d31219ce'
url_test = 'http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_test.csv'
sha1_hash_test = 'fa19780a7b011d9b009e8bff8e99922a8ee2eb90'
folder = './data/kaggle_house'

def check_sha1(filepath, sha1_hash):
    sha1 = hashlib.sha1()
    with open(filepath, 'rb') as f:
        while True:
            data = f.read(1024)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash

def get_data_frame(url, sha1_hash):
    # down load from url
    filename = os.path.join(folder, url.split('/')[-1])
    if not os.path.exists(filename):
        print(f'Downloading {url}...')
        response = requests.get(url, stream=True)

        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
    # hash check
    if sha1_hash:
        assert check_sha1(filename, sha1_hash), "Downloaded file's hash does not match."

    df = pd.read_csv(filename)
    #print(df.head().iloc[:, [0, 1, 2, 3, -3, -2, -1]]) # a quick overview of data
    #print(df.shape)
    return df


class KaggleHouse():
    def __init__(self) -> None:
        self.train = None
        self.val = None
        
    def preprocess(self):
        df_train = get_data_frame(url_train, sha1_hash_train)
        df_test = get_data_frame(url_test, sha1_hash_test)
        print(df_train.shape, df_test.shape)

        label = 'SalePrice'
        features_train = df_train.drop(columns=['Id', label])
        features_test = df_test.drop(columns=['Id'])

        features = pd.concat([features_train, features_test], axis = 0)

        numeric_features = features.dtypes[features.dtypes != 'object'].index

        features[numeric_features] = features[numeric_features].apply(lambda x: (x-x.mean()) / (x.std()))
        features[numeric_features] = features[numeric_features].fillna(0)

        features = pd.get_dummies(features, dummy_na=True)

        self.train = features[:df_train.shape[0]].copy()
        self.train[label] = df_train[label]
        self.val = features[df_train.shape[0]:].copy()

        print(self.train.shape)
        print(self.val.shape)


kh = KaggleHouse()
kh.preprocess()