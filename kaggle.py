import requests
import os
import pandas as pd  # type: ignore
import hashlib

url_train = 'http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_train.csv'
url_test = 'http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_test.csv'
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

def download(url, sha1_hash):
    filename = os.path.join(folder, url.split('/')[-1])
    print(filename)

    if not os.path.exists(filename):
        print(f'Downloading {url}...')
        response = requests.get(url, stream=True)

        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

    if sha1_hash:
        assert check_sha1(filename, sha1_hash), "Downloaded file's hash does not match."

    df = pd.read_csv(filename)
    print(df.head().iloc[:, [0, 1, 2, 3, -3, -2, -1]]) # a quick overview of data
    print(df.shape)

download(url_train, '585e9cc93e70b39160e7921475f9bcd7d31219ce')
#download(url_test, 'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')



