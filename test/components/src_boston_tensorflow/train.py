# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import json
import os
import shutil
import tempfile

import mlflow
import pandas as pd
from azureml.core.run import Run

from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense, Dropout, concatenate

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

def _common_model_generator(feature_number, output_length=1):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(feature_number,)))
    model.add(Dropout(0.25))
    model.add(Dense(output_length, activation='relu', input_shape=(32,)))
    model.add(Dropout(0.5))
    return model

def create_keras_regressor(X, y):
    # create simple (dummy) Keras DNN model for regression
    batch_size = 128
    epochs = 12
    model = _common_model_generator(X.shape[1])
    model.add(Activation('linear'))
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit(X, y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X, y))
    return model

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
    X_train = train_dataset.drop(columns=[args.target_column_name])

    features = args.features

    model = create_keras_regressor(X_train, y_train)

    if args.model_name_suffix < 0:
        suffix = int(time.time())
    else:
        suffix = args.model_name_suffix
    registered_name = "{0}_{1}".format(args.model_base_name, suffix)
    print(f"Registering model as {registered_name}")

    print("Registering via MLFlow")
    mlflow.tensorflow.log_model(
        tensorflow=mlflow_model,
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
