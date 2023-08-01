# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import json
import os
import time

import mlflow
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from azureml.core.run import Run


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", type=str, help="Path to training data")
    parser.add_argument("--target_column_name", type=str, help="Name of target column")
    parser.add_argument("--features", type=json.loads)
    parser.add_argument(
        "--model_info_output_path", type=str, help="Path to write model info JSON"
    )
    parser.add_argument(
        "--model_base_name", type=str, help="Name of the registered model"
    )
    parser.add_argument(
        "--model_name_suffix", type=int, help="Set negative to use epoch_secs"
    )

    # parse args
    args = parser.parse_args()

    # return args
    return args


def _common_pytorch_generator(numCols, num_classes=None):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # Apply layer normalization for stability and perf on wide variety of datasets
            # https://arxiv.org/pdf/1607.06450.pdf
            self.norm = nn.LayerNorm(numCols)
            self.fc1 = nn.Linear(numCols, 100)
            self.fc2 = nn.Dropout(p=0.2)
            if num_classes is None:
                self.fc3 = nn.Linear(100, 3)
                self.output = nn.Linear(3, 1)
            elif num_classes == 2:
                self.fc3 = nn.Linear(100, 2)
                self.output = nn.Sigmoid()
            else:
                self.fc3 = nn.Linear(100, num_classes)
                self.output = nn.Softmax()

        def forward(self, X):
            X = self.norm(X)
            X = F.relu(self.fc1(X))
            X = self.fc2(X)
            X = self.fc3(X)
            return self.output(X)

    return Net()


def _train_pytorch_model(epochs, criterion, optimizer, net, torch_X, torch_y):
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = net(torch_X)
        loss = criterion(out, torch_y)
        loss.backward()
        optimizer.step()
        print("epoch: ", epoch, " loss: ", loss.data.item())
    return net


def create_pytorch_regressor(X, y):
    # create simple (dummy) Pytorch DNN model for regression
    epochs = 12
    if isinstance(X, pd.DataFrame):
        X = X.values
    torch_X = torch.Tensor(X).float()
    torch_y = torch.Tensor(y).float()
    # Create network structure
    net = _common_pytorch_generator(X.shape[1])
    # Train the model
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)
    return _train_pytorch_model(epochs, criterion, optimizer, net, torch_X, torch_y)


def main(args):
    current_experiment = Run.get_context().experiment
    tracking_uri = current_experiment.workspace.get_mlflow_tracking_uri()
    print("tracking_uri: {0}".format(tracking_uri))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(current_experiment.name)

    # Read in data
    print("Reading data")
    train_dataset = pd.read_parquet(args.training_data)

    # Drop the labeled column to get the training set.
    y_train = train_dataset[args.target_column_name]
    X_train = train_dataset[args.features]

    model = create_pytorch_regressor(X_train, y_train)

    if args.model_name_suffix < 0:
        suffix = int(time.time())
    else:
        suffix = args.model_name_suffix
    registered_name = "{0}_{1}".format(args.model_base_name, suffix)
    print(f"Registering model as {registered_name}")

    print("Registering via MLFlow")
    mlflow.pytorch.log_model(
        pytorch_model=model,
        registered_model_name=registered_name,
        artifact_path=registered_name,
    )

    print("Writing JSON")
    dict = {"id": "{0}:1".format(registered_name)}
    output_path = os.path.join(args.model_info_output_path, "model_info.json")
    with open(output_path, "w") as of:
        json.dump(dict, fp=of)


# run script
if __name__ == "__main__":
    # add space in logs
    print("*" * 60)
    print("\n\n")

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
