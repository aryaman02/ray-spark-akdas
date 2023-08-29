from pyspark.sql import SparkSession
from ray.air import session
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster
import ray

from typing import Dict

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler

import torch
from torch import nn
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.nn.init import zeros_
import torch.optim as optim

from ray.train.torch import TorchTrainer, prepare_data_loader, prepare_model
from ray.air.config import ScalingConfig
from torch.utils.data import Dataset, DataLoader, random_split


# class for loading the credit card fraud dataset, eventually train and test DataLoaders will use this!!
class CCDataset(Dataset):
    def __init__(self):
        # read esv file and load row data into variables
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
            data_df = pd.get_dummies(data_df, prefix=[categorical_features[i]], columns=[categorical_features[i]],
                                     drop_first=True)

        # capture the feature data and target labels as numpy arrays
        X = data_df.values[1:, :-1]
        print(X.shape)
        X = X.astype('float32')

        y = data_df['target'].to_numpy()
        y = y[1:].reshape(y.shape[0] - 1, 1)
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y)
        y = ohe.transform(y)

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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
    # instantiate the class to get the dataset, and split it for training and testing
    cc_dataset = CCDataset()
    train_data, test_data = random_split(cc_dataset, [0.05, 0.95])

    return train_data, test_data

def worker_func(config: Dict):
    # set up dataloaders to have multiple workers process the data so that they operate in parallel
    train_data = config["train_data"]
    test_data = config["test_data"]
    batch_size = config["batch_size"]

    worker_batch_size = batch_size // session.get_world_size()

    train_dataloader = DataLoader(train_data, batch_size=worker_batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=worker_batch_size, shuffle=False)

    train_dataloader = prepare_data_loader(train_dataloader)
    test_dataloader = prepare_data_loader(test_dataloader)

    # get size and number of batches of dataloader (represents data shard partitioned for each worker)
    size = len(test_dataloader.dataset) // session.get_world_size()
    num_batches = len(test_dataloader)
    #print(num_batches)

    # hardcode learning rate and num epochs
    lr = config["lr"]
    n_epochs = config["n_epochs"]

    # loss metric and optimizer
    model = Multiclass()
    model = prepare_model(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        # set model in training mode and run through each batch
        model.train()
        for X_tr_batch, y_tr_batch in train_dataloader:
            # forward propagation
            y_pred = model(X_tr_batch)
            loss = loss_fn(y_pred, y_tr_batch)
            # backward propagation
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()

        # set model in evaluation mode and run through the test set
        # for each epoch, compute avg loss and accuracy across the batches in test data partition for each worker

        model.eval()
        
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X_te_batch, y_te_batch in test_dataloader:
                y_pred = model(X_te_batch)
                test_loss += loss_fn(y_pred, y_te_batch).item()
                correct += (torch.argmax(y_pred, 1) == torch.argmax(y_te_batch, 1)).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
    
        print(f"Epoch {epoch} validation: Avg loss={test_loss:.2f}, Accuracy={correct * 100:.1f}%")


if __name__ == "__main__":
    num_workers = 3

    ray.init()

    train_data, test_data = load_and_preprocess_data()


    trainer = TorchTrainer(
        train_loop_per_worker=worker_func,
        train_loop_config={"train_data": train_data, "test_data": test_data,
                           "lr": 0.001, "batch_size": 30, "n_epochs": 20},
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=False),
    )
    result = trainer.fit()




