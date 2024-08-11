import requests
import os
import pandas as pd  # type: ignore
import hashlib
import torch # type: ignore
from torch.utils.data import DataLoader, TensorDataset, random_split # type: ignore

import model
from model import MLP, train_loop
import numpy as np # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore

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

        # conver category to one-hot feature
        features = pd.get_dummies(features, dummy_na=True)
        # Convert bool columns to float
        bool_columns = features.select_dtypes(include=['bool']).columns
        features[bool_columns] = features[bool_columns].astype(float)

        # Ensure all features are numeric
        assert features.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all(), "Not all features are numeric!"

        train_all = features[:df_train.shape[0]].copy()
        train_all[label] = df_train[label]
        val_all = features[df_train.shape[0]:].copy()

        # split traindata to train and val data
        # Shuffle the DataFrame
        train_all = train_all.sample(frac=1).reset_index(drop=True)
        train_size = int(0.7 * len(train_all))
        self.train = train_all[:train_size]
        self.val = train_all[train_size:]
        

    def get_dataloader(self, train=True, batch_size=64):
        data = self.train if train else self.val

        #if train:
        X = data.drop(columns=['SalePrice']).values
        y = data['SalePrice'].values
        y = torch.log(torch.tensor(y, dtype=torch.float32)).reshape(-1, 1)
        #else:
        #    X = data.values
        #    y = None

        X_tensor = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y)
        """
        if train:
            dataset = TensorDataset(X_tensor, y)
        else:
            dataset = TensorDataset(X_tensor)
        """
        return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0)


kaggle_house = KaggleHouse()
kaggle_house.preprocess()

num_epochs = 5
num_inputs = 330
num_hidden = 32
num_outputs = 1

model = MLP(num_inputs, num_hidden, num_outputs)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

train_loader = kaggle_house.get_dataloader(train=True, batch_size=64)
val_loader = kaggle_house.get_dataloader(train=False, batch_size=64)

train_loop(model, train_loader, val_loader, criterion, optimizer, num_epochs=5)
