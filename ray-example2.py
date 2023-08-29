from pyspark.sql import SparkSession
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster
import ray

from typing import Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

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
        # input to hidden layer
        self.hidden = nn.Linear(4, 8)
        kaiming_uniform_(self.hidden.weight, nonlinearity='relu')
        zeros_(self.hidden.bias)
        self.act = nn.ReLU()
        # hidden layer to output
        self.output = nn.Linear(8, 3)
        xavier_uniform_(self.output.weight)
        zeros_(self.output.bias)

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x

def load_data():
    # read data and apply one-hot encoding
    data = pd.read_csv("/Users/aryaman/spark-apps/ray-spark-akdas/iris.csv", header=None)
    X = data.values[:, :-1]
    y = data.iloc[:, 4:]
    # ensure input data is floats
    X = X.astype('float32')
    # label encode target and ensure the values are floats
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y)
    y = ohe.transform(y)
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

    return X_train, X_test, y_train, y_test

def worker_func(config: Dict):
    # prepare model and training parameters
    X_train = config["X_train"]
    X_test = config["X_test"]
    y_train = config["y_train"]
    y_test = config["y_test"]

    batch_size = config["batch_size"]
    batches_per_epoch = len(X_train) // batch_size
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
        ce = loss_fn(y_pred, y_test)
        acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()
        ce = float(ce)
        acc = float(acc)

        print(f"Epoch {epoch} validation: Cross-entropy={ce:.2f}, Accuracy={acc * 100:.1f}%")


if __name__ == "__main__":
    num_workers = 1

    ray.init()

    X_train, X_test, y_train, y_test = load_data()

    trainer = TorchTrainer(
        train_loop_per_worker=worker_func,
        train_loop_config={"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
                           "lr": 1e-3, "batch_size": 5, "n_epochs": 200},
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=False),
    )
    result = trainer.fit()