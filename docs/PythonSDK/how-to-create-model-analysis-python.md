# How to create a Model Analysis Job in AzureML Python SDK
1. Setup
To use responsible AI in AzureML, there are a few pre-requisites that should be setup. The following code allows you to do so.
## Required libraries
```Python
#import required libraries
from azure.ml import MLClient
from azure.ml.entities import CommandJob, Code, PipelineJob, Dataset, InputDatasetEntry
```

## Your Azure ML Details
```Python
#Enter details of your AML workspace
subscription_id = 'ENTER_SUB_ID
resource_group = 'ENTER_RG'
workspace = 'ENTER_WORKSPACE'
```
## Getting a handle to the AML workspace
```Python
#get a handle to the workspace
ml_client = MLClient(subscription_id, resource_group, workspace)
```

3. Initiate Model Analysis
Below is a sample job config - exact config for RAI to be updated
```Python
#define the model analysis setup to run in the pipeline
setup_model_analysis_cmd = 'python score.py --predictions ${{inputs.predictions}} --model ${{inputs.model}} --score_report ${{outputs.score_report}}'
setup_model_analysis_inputs = {
    'predictions': '${{jobs.predict-job.outputs.predictions}}', #use the predictions from predict job so we can score
    'model': '${{jobs.train-job.outputs.model_output}}'} #use the model from the training job
setup_model_analysis_outputs = {'score_report': None}

setup_model_analysis_job = CommandJob(
    code=Code(local_path="./src/score"),
    command = score_cmd,
    inputs = score_job_inputs,
    outputs=score_job_outputs,
    environment = "AzureML-sklearn-0.24-ubuntu18.04-py37-cuda11-gpu:9",
    #compute = "<override with some other compute if needed>"
)
```




