import os

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from pyspark.sql import SparkSession
from ray.air import session, Checkpoint
from ray.tune.schedulers import ASHAScheduler
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
from torch.utils.data import Dataset, random_split, DataLoader

from ray import tune


# class for loading the credit card fraud dataset, eventually train and test DataLoaders will use this!!
class CCDataset(Dataset):
    def __init__(self):
        # read esv file and load row data into variables
        data_df = pd.read_csv('/Users/aryaman/spark-apps/ray-spark-akdas/creditcard.csv')

        X_orig = data_df.loc[:, data_df.columns != 'Class']
        y_orig = data_df['Class']

        over = SMOTE(sampling_strategy=0.03)
        under = RandomUnderSampler(sampling_strategy=0.3)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        X_res, y_res = pipeline.fit_resample(X_orig, y_orig)

        new_df = pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res)], axis=1)

        # normalize the continuous features
        sc = MinMaxScaler()
        X = new_df.values[1:, :-1]
        X = X.astype('float32')
        X = sc.fit_transform(X)

        # capture the feature data and target labels as numpy arrays
        print(X.shape)

        y = new_df['Class'].to_numpy()
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
    def __init__(self, l1=128, l2=32, l3=8):
        super().__init__()
        # input to first hidden layer
        self.hidden1 = nn.Linear(30, l1)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        zeros_(self.hidden1.bias)
        self.activation1 = nn.ReLU()

        # more hidden layers
        self.hidden2 = nn.Linear(l1, l2)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        zeros_(self.hidden2.bias)
        self.activation2 = nn.ReLU()

        self.hidden3 = nn.Linear(l2, l3)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        zeros_(self.hidden3.bias)
        self.activation3 = nn.ReLU()

        # last hidden layer to output
        self.output = nn.Linear(l3, 2)
        xavier_uniform_(self.output.weight)
        zeros_(self.output.bias)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.activation1(x)
        x = self.hidden2(x)
        x = self.activation2(x)
        x = self.hidden3(x)
        x = self.activation3(x)
        x = self.output(x)
        return x

def load_and_preprocess_data():
    # instantiate the class to get the dataset, and split it for training and testing
    cc_dataset = CCDataset()
    train_data, test_data = random_split(cc_dataset, [0.7, 0.3])

    return train_data, test_data

def training_func_tune(config: Dict):
    # set up training dataloader to have multiple workers process the data so that they operate in parallel
    train_data = config["train_data"]
    batch_size = config["batch_size"]

    worker_batch_size = batch_size // session.get_world_size()

    train_dataloader = DataLoader(train_data, batch_size=worker_batch_size, shuffle=True)

    train_dataloader = prepare_data_loader(train_dataloader)

    # hardcode learning rate and maximum number of epochs per trial
    lr = config["lr"]
    max_n_epochs = config["max_n_epochs"]

    # loss metric and optimizer
    model = Multiclass(config["l1"], config["l2"], config["l3"])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # To restore a checkpoint, use `session.get_checkpoint()`.
    # load the state of the deep neural network model from an existing trial so that model is trained in the right direction
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    model = prepare_model(model)

    for epoch in range(max_n_epochs):
        # set model in training mode and run through each batch
        model.train()
        running_loss = 0.0
        epoch_steps = 0
        for X_tr_batch, y_tr_batch in train_dataloader:
            # forward propagation
            y_pred = model(X_tr_batch)
            loss = loss_fn(y_pred, y_tr_batch)
            # backward propagation
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # increment accumulated loss after training each batch of data
            running_loss += loss.item()
            epoch_steps += 1
        # save the current state of model in checkpoint directory for later retrieval
        # we use minimum training loss as a tune metric for early stopping of trial and determining best trial
        os.makedirs("my_model", exist_ok=True)
        torch.save(
            (model.state_dict(), optimizer.state_dict()), "my_model/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("my_model")
        session.report({"training_loss": (running_loss / epoch_steps)}, checkpoint=checkpoint)

    print("Finished Training")

def test_best_model_func(config: Dict):
    # set up test dataloader to have multiple workers process the data so that they operate in parallel
    test_data = config["test_data"]
    worker_batch_size = 10
    test_dataloader = DataLoader(test_data, batch_size=worker_batch_size, shuffle=True)
    test_dataloader = prepare_data_loader(test_dataloader)

    # get size of test dataset shard for each worker
    size = len(test_dataloader.dataset) // session.get_world_size()

    # get best model from the tuning process, and load its state
    best_trained_model = Multiclass(config["l1"], config["l2"], config["l3"])
    best_trained_model.load_state_dict(config["model_state_dict"], strict=False)

    best_trained_model = prepare_model(best_trained_model)
    # set model in evaluation mode and run through the test set
    best_trained_model.eval()

    correct = 0
    with torch.no_grad():
        for X_te_batch, y_te_batch in test_dataloader:
            y_pred = best_trained_model(X_te_batch)
            correct += (torch.argmax(y_pred, 1) == torch.argmax(y_te_batch, 1)).type(torch.float).sum().item()

    print("Best trial test set accuracy: {}".format(correct / size))


if __name__ == "__main__":
    # load the data, and set up search space for hyperparameter tuning
    train_data, test_data = load_and_preprocess_data()
    max_num_epochs = 80
    config = {
        "train_loop_config": {
            "train_data": train_data,
            "test_data": test_data,
            "max_n_epochs": max_num_epochs,
            "batch_size": tune.choice([20, 40, 60, 80]),
            "l1": tune.sample_from(lambda _: 2 ** np.random.randint(6, 10)),
            "l2": tune.sample_from(lambda _: 2 ** np.random.randint(5, 9)),
            "l3": tune.sample_from(lambda _: 2 ** np.random.randint(3, 7)),
            "lr": tune.loguniform(1e-4, 1e-1)
        },
        "scaling_config": ScalingConfig(
            num_workers=1,
            trainer_resources={"CPU": 1},
            use_gpu=False
        ),
    }
    # use a torch trainer as trainable for ray tune
    trainer = TorchTrainer(training_func_tune)
    # use this scheduler for stopping trials early that perform poorly on training loss metric
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    num_trials_tune = 15 # max number of trials for hyperparameter tuning

    tuner = tune.Tuner(trainer,
        tune_config=tune.TuneConfig(
            metric="training_loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_trials_tune,
            max_concurrent_trials=15
        ),
        param_space=config,
    )
    results = tuner.fit() # calls ray.init() under the hood!
    # get information about the best trial
    best_result = results.get_best_result("training_loss", "min")

    print("Best trial config: {}".format(best_result.config))

    # load the state of model (weight and bias values) from best_result
    loaded_checkpoint = best_result.checkpoint
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))

    test_config = {
        "test_data": best_result.config["train_loop_config"]["test_data"],
        "l1": best_result.config["train_loop_config"]["l1"],
        "l2": best_result.config["train_loop_config"]["l2"],
        "l3": best_result.config["train_loop_config"]["l3"],
        "model_state_dict": model_state
    }

    test_trainer = TorchTrainer(
        train_loop_per_worker=test_best_model_func,
        train_loop_config=test_config,
        scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
    )
    # get accuracy of best model generated from ray tune
    result = test_trainer.fit()