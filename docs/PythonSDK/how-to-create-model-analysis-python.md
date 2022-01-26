# How to create a Model Analysis Job in AzureML Python SDK

To use responsible AI in AzureML, there are a few pre-requisites that should be setup. The following code allows you to do so.

## Setup

First, you need to download the `config.json` file for your target workspace from Azure.
This needs to be placed in the directory from where you run `python` (or `jupyter`).

### Required libraries
```Python
#import required libraries
from azure.ml import MLClient
from azure.ml.entities import CommandJob, Code, PipelineJob, Dataset, InputDatasetEntry
```

### Your Azure ML Details
```Python
# Obtain a client
from azure.ml import MLClient
from azure.identity import DefaultAzureCredential
ml_client = MLClient.from_config(credential=DefaultAzureCredential(exclude_shared_token_cache_credential=True),
                     logging_enable=True)
```

## Constructing jobs for our pipeline
Each job requires a specification of inputs, outputs, and task to be performed to convert those inputs to outputs. Each section below specifies a job that will be connected in our resulting pipeline.

### Set your global pipeline inputs
Now we will start to construct our pipeline using jobs. To start, we first want to set any inputs that are used across the entire pipeline, from inputting the data to training the model to conducting the analysis. For this sample we will save our target column, training data and testing data as global pipeliine inputs. In this example, the classic 'Adult' loaded into our workspace, presplit into train and test subsets, and stored in a Parquet file. We will be using version 1 of this dataset:
```Python
pipeline_inputs = { 
    'target_column_name': 'income',
    'my_training_data': InputDatasetEntry(dataset=f"Adult_Train_PQ:1"),
    'my_test_data': InputDatasetEntry(dataset=f"Adult_Test_PQ:1")
}
```
### Create a training job and register your model
The next job to create in our pipeline will be to train a simple logistic regression model using the data imported in the previous step and identifying which column you would like to predict on in the target column.
```Python
# Specify the training job
train_job_inputs = {
    'target_column_name': '${{inputs.target_column_name}}',
    'training_data': '${{inputs.my_training_data}}',
}
train_job_outputs = {
    'model_output': None
}

train_job = ComponentJob(
    component=f"TrainLogisticRegressionForRAI:{component_dataset_version_suffix}",
    inputs = train_job_inputs,
    outputs=train_job_outputs
)

# Use the output from the training job to register the model in your workspace
register_job_inputs = {
    'model_input_path': '${{jobs.train-model-job.outputs.model_output}}',
    'model_base_name': 'notebook_registered_logreg',
}
register_job_outputs = {
    'model_info_output_path': None
}
register_job = ComponentJob(
    component=f"RegisterModel:{version_string}",
    inputs=register_job_inputs,
    outputs=register_job_outputs
)
```

### Create a model analysis job
Create a model analysis job by using inputs of your training data, test data, and the registered model that was output from your training job.

Below is a sample job config for configuring the setup model analysis component. In this component, you will need to enter your datastore name.
```Python
#define the model analysis setup to run in the pipeline

setup_model_analysis_inputs = {
     'title': 'My Model Analysis',
     'model_info_path': '${{jobs.register-model-job.outputs.model_info_output_path}}', #this will take the model that you registered in the previous step as input to your model analysis
    'train_dataset': '${{inputs.my_training_data}}',
    'test_dataset': '${{inputs.my_test_data}}',
    'target_column_name': '${{inputs.target_column_name}}',
    'X_column_names': '["Age", "Workclass", "Education-Num", "Marital Status", "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss", "Hours per week", "Country"]',
            
    'datastore_name': 'YOURDATASTORENAME', #replace with your datastore name
    'categorical_column_names': '["Race", "Sex", "Workclass", "Marital Status", "Country", "Occupation"]',

setup_model_analysis_outputs = {'score_report': None}

setup_model_analysis_job = CommandJob(
    component=f"AzureMLModelAnalysis:{version_string}",
    inputs=create_ma_inputs,
    outputs=create_ma_outputs
)
```

### Add desired model analysis components
After setting up your model analysis component, choose which functionalities you would like to be computed and shown in your dashboard. We currently support
- Explainability
- Causal Analysis
- Counterfactual Analysis
- Error Analysis

For this example, we will be showing the explain component. Checkout [the components document]() to determine the other components other components you may like to include.

```Python
#define which responsible AI components to use in your model analysis
 explain_inputs = {
            'comment': 'Insert text here',
            'model_analysis_info': '${{jobs.create-ma-job.outputs.model_analysis_info}}'
}
explain_job = ComponentJob(
    component=f"AzureMLModelAnalysisExplanation:{version_string}",
    inputs = explain_inputs
)
```
## Create a pipeline with the specified jobs
### Create a pipeline with the jobs defined above
```Python
# lets create the pipeline
pipeline_job = PipelineJob(
    experiment_name=f"Classification_from_Python_{version_string}",
    description="My first Model analysis",
    jobs = {
        'train-model-job': train_job,
        'register-model-job': register_job,
        'create-ma-job': create_ma_job,
        'explain-ma-job': explain_job,
    },
    inputs=pipeline_inputs,
    outputs=train_job_outputs,
    compute="cpucluster"
)

```
### Submit the pipeline job
```Python
#submit the pipeline job
submit_and_wait(ml_client, pipeline_job)
```

### Sample exploring....
```Python
from azure.ml import dsl, load_component, MLClient
from contoso.components import (
    data_ingestion,
    data_preprocess,
    train_cool_model,
    evaluate_model,
    rai_dashboard,
    deploy,
)


# This is the only "pointer" to an authenticated object.
ml_client = MLClient("420e53f3-bf59-43cd-9595-4259eeceb54c", "my-rg", "my-ws")


# Components may be obtained directly from local YAML files or workspace clients.
alpha_component = load_component(yaml_file="path/to/spec.yaml")
beta_component = load_component(ml_client, "beta-component")


@dsl.pipeline(name="object classification", default_compute_target="cpu-cluster")
def object_classification(
    num_images=100,
    image_dim=200,
    num_epochs=10,
    batch_size=16,
    learning_rate=0.001,
    momentum=0.9,
):
    data = data_ingestion(num_images=num_images)
    preprocessed_data = data_preprocess(
        data.outputs.raw_data, image_dimension=image_dim
    )
    training = train_cool_model(
        # Both attribute and dictionary style output access work.
        training=preprocessed_data.outputs["train_dir"],
        validation=preprocessed_data.outputs.valid_dir,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        momentum=momentum,
    )
    training.compute.target = "gpu-cluster"

    evaluate_model(
        model=training.outputs.model,
        accuracy=training.outputs.accuracy_file,
        test=training.outputs.test_model,
    )
    rai = rai_dashboard(
        model = training.outputs.model
        test = training.outputs.test_model
        training=preprocessed_data.outputs["train_dir"],
        explain = true
    )
    deploy(training.outputs.model)


pipeline = object_classification()

job = pipeline.submit(ml_client, description="my cool pipeline")
print(f"Submitted pipeline: {job.get_portal_url()}")
```


