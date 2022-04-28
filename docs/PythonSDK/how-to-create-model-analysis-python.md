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

One thing we need to specify is the component version we are using.
This will have been set when the components were registered into the workspace.
For this sample, we shall assume that the components were also registered with a version of 1:
```python
version_string = "1"
```

### Set your global pipeline inputs
Now we will start to construct our pipeline using jobs. To start, we first want to set any inputs that are used across the entire pipeline, from inputting the data to training the model to conducting the analysis. For this sample we will save our target column, training data and testing data as global pipeliine inputs. In this example, the classic 'Adult' loaded into our workspace, presplit into train and test subsets, and stored in a Parquet file. We will be using version 1 of this dataset:

```python
pipeline_inputs = { 
    'target_column_name': 'income',
    'my_training_data': InputDatasetEntry(dataset=f"adult_train_pq:1"),
    'my_test_data': InputDatasetEntry(dataset=f"adult_test_pq:1")
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
    component=f"train_logistic_regression_for_rai:{component_dataset_version_suffix}",
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
    component=f"register_model:{version_string}",
    inputs=register_job_inputs,
    outputs=register_job_outputs
)
```

### Create an RAIInsights constructor job

Creating `RAIInsights` through AzureML is a multi-step process.
First, we use a 'constructor' component to configure all the common settings for a given analysis.
These are things such as the model we wish to analyse, the datasets to use, and settings which are common to all analyses we might perform.

Below is a sample job config for configuring the `RAIInsight` constructor component.
```Python
create_rai_inputs = {
    "title": "Run built from Python",
    "task_type": "classification",
    "model_info_path": "${{jobs.register-model-job.outputs.model_info_output_path}}",
    "train_dataset": "${{inputs.my_training_data}}",
    "test_dataset": "${{inputs.my_test_data}}",
    "target_column_name": "${{inputs.target_column_name}}",
    "categorical_column_names": '["Race", "Sex", "Workclass", "Marital Status", "Country", "Occupation"]',
}
create_rai_outputs = {"rai_insights_dashboard": None}
create_rai_job = ComponentJob(
    component=f"rai_insights_constructor:{version_string}",
    inputs=create_rai_inputs,
    outputs=create_rai_outputs,
)
```

### Add desired RAIInsights components
After setting up your constructor component, choose which functionalities you would like to be computed and shown in your dashboard. We currently support
- Explainability
- Causal Analysis
- Counterfactual Analysis
- Error Analysis

For this example, we will be showing the explain component. Checkout [the components document]() to determine the other components other components you may like to include.

```Python
explain_inputs = {
    "comment": "This is our explanation",
    "rai_insights_dashboard": "${{jobs.create-rai-job.outputs.rai_insights_dashboard}}",
}
explain_outputs = {"explanation": None}
explain_job = ComponentJob(
    component=f"rai_insights_explanation:{version_string}",
    inputs=explain_inputs,
    outputs=explain_outputs,
)
```

### Gather the RAIInsights

Finally, we have a 'gather' job which can assemble the model insights we have computed into a single dashboard.
This also needs to be provided with the original constructor output:

```python
# Configure the gather component
gather_inputs = {
    "constructor": "${{jobs.create-rai-job.outputs.rai_insights_dashboard}}",
    "insight_1": "${{jobs.explain-rai-job.outputs.explanation}}",
}
gather_outputs = {"dashboard": None, "ux_json": None}
gather_job = ComponentJob(
    component=f"rai_insights_gather:{version_string}",
    inputs=gather_inputs,
    outputs=gather_outputs,
)
```
Up to four insights (one of each type) can be gathered, but in this case we just have the explanation.

### Create a pipeline with the jobs defined above

Now that we have defined all the components we need, we can assemble them into
an AzureML pipeline:

```python
pipeline_job = PipelineJob(
    experiment_name=f"classification_pipeline_from_python_{version_string}",
    description="Python submitted Adult",
    jobs={
        "train-model-job": train_job,
        "register-model-job": register_job,
        "create-rai-job": create_rai_job,
        "explain-rai-job": explain_job,
        "gather-job": gather_job,
    },
    inputs=pipeline_inputs,
    outputs=train_job_outputs,
    compute="cpucluster", # This is the name of our AMLCompute
)
```

### Submit the pipeline

Submitting the job is fairly straightforward:

```python
created_job = ml_client.jobs.create_or_update(pipeline_job)
```

Of course, it will take a few minutes to run (especially if compute nodes have to be spun up).
Progress can be monitored via:
```python
latest_job = ml_client.jobs.get(created_job.name)
print(latest_job.status)
```
(or by looking in the AzureML portal).


## To be deleted?

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


