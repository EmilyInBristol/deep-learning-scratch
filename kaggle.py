import requests
import os
import pandas as pd  # type: ignore
import hashlib
import torch # type: ignore
from torch.utils.data import DataLoader, TensorDataset, random_split # type: ignore

import model
from model import MLP, LinearRegression, train_loop, train_model, draw
import numpy as np # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import logging
# Set up logging to print debug messages to the console
logging.basicConfig(level=logging.INFO)

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
        logging.info(f'Downloading {url}...')
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
    logging.info(f'DataFrame shape: {df.shape}')
    return df

class KaggleHouse():
    def __init__(self, train, val, batch_size=64) -> None:
        self.raw_train = train
        self.raw_val = val
        self.train = None
        self.val = None
        self.batch_size = batch_size
        
    def preprocess(self):
        # preprocess the features
        label = 'SalePrice'
        features_train = self.raw_train.drop(columns=['Id', label])
        if label not in self.raw_val.columns:
            features_test = self.raw_val.drop(columns=['Id'])
        else:
            features_test = self.raw_val.drop(columns=['Id', label])
        logging.info(f'features_train shape:{features_train.shape} features_test shape:{features_test.shape}')
        features = pd.concat([features_train, features_test], axis = 0)
        # numeric values nomalize
        numeric_features = features.dtypes[features.dtypes != 'object'].index
        features[numeric_features] = features[numeric_features].apply(lambda x: (x-x.mean()) / (x.std()))
        features[numeric_features] = features[numeric_features].fillna(0)
        # convert category to one-hot feature
        features = pd.get_dummies(features, dummy_na=True)
        # convert bool columns to float
        bool_columns = features.select_dtypes(include=['bool']).columns
        features[bool_columns] = features[bool_columns].astype(float)

        # Ensure all features are numeric
        assert features.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all(), "Not all features are numeric!"

        self.train = features[:self.raw_train.shape[0]].copy()
        self.train[label] = self.raw_train[label]
        self.val = features[self.raw_train.shape[0]:].copy()
        if label in self.raw_val.columns:
            self.val[label] = self.raw_val[label]

        logging.info(f'preprocess done, train.shape: {self.train.shape}, val.shape: {self.val.shape}')

    def get_dataloader(self, train=True):
        data = self.train if train else self.val    
        label = 'SalePrice'

        if not train and label not in data:
            X = data.values
            y = None
        else:
            X = data.drop(columns=['SalePrice']).values
            y = data['SalePrice'].values
            y = torch.log(torch.tensor(y, dtype=torch.float32)).reshape(-1, 1)
        
        X = torch.tensor(X, dtype=torch.float32)
        
        dataset = None
        if not train and label not in data:
            dataset = TensorDataset(X)
        else:
            dataset = TensorDataset(X, y)
       
        return DataLoader(dataset, self.batch_size, shuffle=train, num_workers=0)

def k_fold_data(data, k):
    fold_size = data.shape[0] // k
    rets = []
    for i in range(k):
        idx = range(i * fold_size, (i+1) * fold_size)
        i_train = data.drop(index=idx)
        i_val = data.loc[idx]
        rets.append(KaggleHouse(i_train, i_val, 64))
    return rets

if __name__ == '__main__':

    kaggle_house_train = get_data_frame(url_train, sha1_hash_train)
    kaggle_house_test = get_data_frame(url_test, sha1_hash_test)

    num_epochs = 5
    num_inputs = 330
    num_hidden = 32
    num_outputs = 1

    avg_val_loss = []
    models = []
    for i, kh in enumerate(k_fold_data(kaggle_house_train, 5)):
        model = LinearRegression()
        optimizer = model.configure_optimizers()
        criterion = nn.MSELoss()
        kh.preprocess()
        train_loader= kh.get_dataloader(train=True)
        val_loader = kh.get_dataloader(train=False)
        train_loss, val_loss = train_loop(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)        

        avg_val_loss.append(val_loss[-1])
        models.append(model)

    print(sum(avg_val_loss)/len(avg_val_loss))
    print(avg_val_loss)

    data = KaggleHouse(kaggle_house_train, kaggle_house_test)
    data.preprocess()

    preds = [model(torch.tensor(data.val.values.astype(float), dtype=torch.float32))
         for model in models]
    # Taking exponentiation of predictions in the logarithm scale
    ensemble_preds = torch.exp(torch.cat(preds, 1)).mean(1)
    submission = pd.DataFrame({'Id':data.raw_val.Id,
                            'SalePrice':ensemble_preds.detach().numpy()})
    submission.to_csv('submission.csv', index=False)
