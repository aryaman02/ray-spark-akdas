from pyspark.sql import SparkSession
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster
import ray

from typing import Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler

import torch
from torch import nn
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.nn.init import zeros_
import torch.optim as optim

from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig

# Define model
class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        # input to first hidden layer
        self.hidden1 = nn.Linear(51, 128)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        zeros_(self.hidden1.bias)
        self.activation1 = nn.ReLU()

        # more hidden layers
        self.hidden2 = nn.Linear(128, 32)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        zeros_(self.hidden2.bias)
        self.activation2 = nn.ReLU()

        self.hidden3 = nn.Linear(32, 8)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        zeros_(self.hidden3.bias)
        self.activation3 = nn.ReLU()

        """
        self.hidden4 = nn.Linear(32, 8)
        kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        zeros_(self.hidden4.bias)
        self.activation4 = nn.ReLU()
        """

        # last hidden layer to output
        self.output = nn.Linear(8, 2)
        xavier_uniform_(self.output.weight)
        zeros_(self.output.bias)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.activation1(x)
        x = self.hidden2(x)
        x = self.activation2(x)
        x = self.hidden3(x)
        x = self.activation3(x)
        #x = self.hidden4(x)
        #x = self.activation4(x)
        x = self.output(x)
        return x

def load_and_preprocess_data():
    # read the csv data
    data_df = pd.read_csv('/Users/aryaman/spark-apps/ray-spark-akdas/credit_card_fraud.csv')

    # drop features that are unnecessary or might correlate with one another
    drop_features = ['CNT_CHILDREN', 'DAYS_BIRTH', 'FLAG_WORK_PHONE', 'CNT_FAM_MEMBERS', 'QUANTIZED_WORK_YEAR']
    for i in range(len(drop_features)):
        data_df.drop(drop_features[i], axis=1, inplace=True)

    # normalize the continuous features
    continuous_features = ['AMT_INCOME_TOTAL', 'DAYS_EMPLOYED']
    sc = MinMaxScaler()
    data_df[continuous_features] = sc.fit_transform(data_df[continuous_features])

    # normalize the categorical features (scale them uniformly)
    categorical_features = ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
                             'OCCUPATION_TYPE', 'QUANTIZED_INC',
                             'QUANTIZED_AGE']

    le = LabelEncoder()
    for i in range(len(categorical_features)):
        data_df[categorical_features[i]] = le.fit_transform(data_df[categorical_features[i]])

    for i in range(len(categorical_features)):
        data_df = pd.get_dummies(data_df, prefix=[categorical_features[i]], columns=[categorical_features[i]], drop_first=True)

    # capture the feature data and target labels as numpy arrays
    X = data_df.values[1:, :-1]
    print(X.shape)
    X = X.astype('float32')

    y = data_df['target'].to_numpy()
    y = y[1:].reshape(y.shape[0]-1, 1)
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y)
    y = ohe.transform(y)


    X = torch.from_numpy(X)
    y = torch.from_numpy(y)


    # split data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.04, shuffle=True)

    print(y_test)

    return X_train, X_test, y_train, y_test

def worker_func(config: Dict):
    # prepare model and training parameters
    X_train = config["X_train"]
    X_test = config["X_test"]
    y_train = config["y_train"]
    y_test = config["y_test"]

    batch_size = config["batch_size"]
    batches_per_epoch = len(X_train) // batch_size
    print(X_train.shape)
    lr = config["lr"]
    n_epochs = config["n_epochs"]

    # loss metric and optimizer
    model = Multiclass()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        # set model in training mode and run through each batch
        model.train()
        for i in range(batches_per_epoch):
            # take a batch
            start = i * batch_size
            X_batch = X_train[start:start + batch_size]
            y_batch = y_train[start:start + batch_size]
            # forward propagation
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward propagation
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()

        # set model in evaluation mode and run through the test set
        model.eval()
        y_pred = model(X_test)
        #print(y_pred)
        #print(y_test)
        ce = loss_fn(y_pred, y_test)
        acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()
        ce = float(ce)
        acc = float(acc)

        print(f"Epoch {epoch} validation: Cross-entropy={ce:.2f}, Accuracy={acc * 100:.1f}%")


if __name__ == "__main__":
    num_workers = 1

    ray.init()

    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    trainer = TorchTrainer(
        train_loop_per_worker=worker_func,
        train_loop_config={"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
                           "lr": 0.001, "batch_size": 50, "n_epochs": 150},
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=False),
    )
    result = trainer.fit()