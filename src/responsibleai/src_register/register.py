# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import json
import os
import tempfile
import time

import mlflow
import mlflow.sklearn

# Based on example:
# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-cli
# which references
# https://github.com/Azure/azureml-examples/tree/main/cli/jobs/train/lightgbm/iris


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--model_input_path", type=str, help="Path to input model")
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


def main(args):
    tracking_uri = mlflow.get_tracking_uri()
    print("tracking_uri: {0}".format(tracking_uri))
    print("MLflow version: {0}".format(mlflow.__version__))

    print("Loading model")
    mlflow_model = mlflow.sklearn.load_model(args.model_input_path)

    if args.model_name_suffix < 0:
        suffix = int(time.time())
    else:
        suffix = args.model_name_suffix
    registered_name = "{0}_{1}".format(args.model_base_name, suffix)
    print(f"Registering model as {registered_name}")

    print("Logging model via MLFlow using save_model approach")
    # Use save_model and create_model_version with file:// URI to let Azure ML handle the upload
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = os.path.join(temp_dir, registered_name)
        mlflow.sklearn.save_model(mlflow_model, model_dir)

        # Use the older model registry API directly to avoid logged-models search
        from mlflow.tracking import MlflowClient
        client = MlflowClient()

        try:
            # Try to create the registered model (will fail if it already exists)
            client.create_registered_model(registered_name)
            print(f"Created new registered model: {registered_name}")
        except Exception as e:
            print(f"Registered model {registered_name} already exists: {e}")

        # Create a new version of the model using file:// URI
        # Azure ML will handle the upload and generate the proper azureml:// URI
        file_uri = f"file://{model_dir}"
        print("Registering model with file_uri: {0}".format(file_uri))

        model_version = client.create_model_version(
            name=registered_name,
            source=file_uri
        )
        print(f"Created model version {model_version.version} for {registered_name}")

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
