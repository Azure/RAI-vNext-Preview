# How to create a Model Analysis Job in AzureML Python SDK
1. Setup
To use responsible AI in AzureML, there are a few pre-requisites that should be setup. The following code allows you to do so.
## Setup
### Required libraries
```Python
#import required libraries
from azure.ml import MLClient
from azure.ml.entities import CommandJob, Code, PipelineJob, Dataset, InputDatasetEntry
```

### Your Azure ML Details
```Python
#Enter details of your AML workspace
subscription_id = 'ENTER_SUB_ID'
resource_group = 'ENTER_RG'
workspace = 'ENTER_WORKSPACE'
```
### Getting a handle to the AML workspace
```Python
#get a handle to the workspace
ml_client = MLClient(subscription_id, resource_group, workspace)
```

### Connect your dataset
```Python
#Sample
```
## Constructing jobs for our pipeline
Each job requires a specification of inputs, outputs, and task to be performed to convert those inputs to outputs. Each section below specifies a job that will be connected in our resulting pipeline.

### Set your global pipeline inputs
Now we will start to construct our pipeline using jobs. To start, we first want to set any inputs that are used across the entire pipeline, from inputting the data to training the model to conducting the analysis. For this sample we will save our target column, training data and testing data as global pipeliine inputs.
```Python
pipeline_inputs = { 
    'target_column_name':'income',
    'my_training_data': InputDatasetEntry(dataset=f"Adult_Train_PQ:{component_dataset_version_suffix}"),
    'my_test_data': InputDatasetEntry(dataset=f"Adult_Test_PQ:{component_dataset_version_suffix}")
}
```
### Create a training job and register your model
The next job to create in our pipeline will be to train a simple logistic regression model using the data imported in the previous step and identifying which column you would like to predict on in the target column.
```Python
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
```

### Create a model analysis job
Create a model analysis job by using inputs of your training data, test data, and the registered model that was output from your training job.

Below is a sample job config for configuring the setup model analysis component. In this component, you will need to enter your datastore name.
```Python
#define the model analysis setup to run in the pipeline
setup_model_analysis_cmd = 'python score.py --predictions ${{inputs.predictions}} --model ${{inputs.model}} --score_report ${{outputs.score_report}}'
setup_model_analysis_inputs = {
     'title': 'My Model Analysis',
    'task_type': 'classification',
    'model_info_path': '${{jobs.register-model-job.outputs.model_info_output_path}}', #use the model from the training job
    'train_dataset': '${{inputs.my_training_data}}',
    'test_dataset': '${{inputs.my_test_data}}',
    'target_column_name': '${{inputs.target_column_name}}',
    'X_column_names': '["Age", "Workclass", "Education-Num", "Marital Status", "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss", "Hours per week", "Country"]',
    'datastore_name': 'YOURDATASTORENAME', #replace with your datastore name
    'categorical_column_names': '["Race", "Sex", "Workclass", "Marital Status", "Country", "Occupation"]',
setup_model_analysis_outputs = {'score_report': None}

setup_model_analysis_job = CommandJob(
    code=Code(local_path="./src/score"),
    command = setup_model_analysis_cmd,
    inputs = setup_model_analysis_inputs,
    outputs=setup_model_analysis_outputs,
    environment = "AzureML-sklearn-0.24-ubuntu18.04-py37-cuda11-gpu:9",
    #compute = "<override with some other compute if needed>"
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
error_analysis_cmd = 'python score.py --predictions ${{inputs.predictions}} --model ${{inputs.model}} --score_report ${{outputs.score_report}}'
error_analysis_inputs = {
    'predictions': '${{jobs.predict-job.outputs.predictions}}', #use the predictions from predict job so we can score
    'model': '${{jobs.train-job.outputs.model_output}}'} #use the model from the training job
error_analysis_outputs = {'score_report': None}

error_analysis_job = CommandJob(
    code=Code(local_path="./src/score"),
    command = error_analysis_cmd,
    inputs = error_analysis_inputs,
    outputs=error_analysis_outputs,
    environment = "AzureML-sklearn-0.24-ubuntu18.04-py37-cuda11-gpu:9",
    #compute = "<override with some other compute if needed>"
)
```
## Create a pipeline with the specified jobs
### Create a pipeline with the jobs defined above
```Python
# lets create the pipeline
pipeline_job = PipelineJob(
    description = 'nyc-taxi-pipeline-example',
    jobs= {
        'train-model-job': train_job,
        'register-model-job': register_job,
        'setup_model_analysis_job':setup_model_analysis_job, 
        'error_analysis_job': error_analysis_job}, #add all the jobs into this pipeline
    inputs= pipeline_job_inputs, #top level inputs to the pipeline
    outputs=prep_job_outputs,
    compute = "gpu-cluster"
)

```
### Submit the pipeline job
```Python
#submit the pipeline job
returned_job = ml_client.jobs.create_or_update(pipeline_job)
#get a URL for the status of the job
returned_job.services["Studio"].endpoint
```



