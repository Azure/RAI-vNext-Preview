# How to create a Model Analysis Job in AzureML Python SDK

This document provides an overview of how to construct a Responsible AI analysis pipeline in AzureML.
For fully working examples, please [see the notebooks](../../examples/notebooks/).

## Setup

First, you need to download the `config.json` file for your target workspace from Azure.
This needs to be placed in the directory from where you run `python` (or `jupyter`).

### Your Azure ML Details

To interact with AzureML, we need an instance of the `MLClient` class:
```Python
# Obtain a client
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
ml_client = MLClient.from_config(
    credential=DefaultAzureCredential(exclude_shared_token_cache_credential=True),
    logging_enable=True
)
```
If there are authentication issues, the error message from `DefaultAzureCredential()` should contain links to documentation
to help resolve them.

## Constructing jobs for our pipeline
Each job requires a specification of inputs, outputs, and task to be performed to convert those inputs to outputs. Each section below specifies a job that will be connected in our resulting pipeline.

One thing we need to specify is the component version we are using.
This will have been set when the components were registered into the workspace.
For this sample, we shall assume that the components were also registered with a version of 1:
```python
version_string = "1"
```

### Creating the pipeline inputs

We need two inputs for our pipeline - the training and test datasets.
Here, we assume that they are already available in the workspace with a version of 1.

```python
from azure.ai.ml import Input

diabetes_train_pq = Input(
    type="uri_file",
    path=f"diabetes_decision_train_pq:1",
    mode="download"
)
diabetes_test_pq = Input(
    type="uri_file",
    path=f"diabetes_decision_test_pq:1",
    mode="download"
)
```


### Loading the components

Pipelines are built from components, so we need to load them next:
```python
fetch_model_component = ml_client.components.get(
    name='fetch_registered_model', version=version_string
)

rai_constructor_component = ml_client.components.get(
    name="rai_insights_constructor", version=version_string
)

rai_causal_component = ml_client.components.get(
    name="rai_insights_causal", version=version_string
)

rai_counterfactual_component = ml_client.components.get(
    name="rai_insights_counterfactual", version=version_string
)

rai_gather_component = ml_client.components.get(
    name="rai_insights_gather", version=version_string
)

rai_scorecard_component = ml_client.components.get(
    name="rai_score_card", version=version_string
)
```

### Configuring the scorecard

We're going to create a pipeline which includes a scorecard based on the Responsible AI analysis.
A JSON file is used to specify the scorecard.
We need to create this first, so that it can be passed into the pipeline:

```python
import json

score_card_config_dict = {
    "Model": {
        "ModelName": "Diabetes disease progression measure",
        "ModelType": "Regression",
        "ModelSummary": "This model provides a quantitative measure of disease progression one year after baseline"
    },
    "Metrics" :{
        "mean_absolute_error": {
            "threshold": "<=5"
        },
        "mean_squared_error": {}
    }
}

score_card_config_filename = "rai_diabetes_decision_score_card_config.json"

with open(score_card_config_filename, 'w') as f:
    json.dump(score_card_config_dict, f)
```

We then make the JSON file into another `Input`:
```python
score_card_config_path = Input(
    type="uri_file",
    path=score_card_config_filename,
    mode="download"
)
```

### Creating the pipeline

We now create the pipeline itself using the DSL created for this purpose.
We assume that the model has previously been registered in AzureML, and use the
'Fetch Registered Model' component to retrieve it.

The Responsible AI components often require complex inputs, such as lists of
column names.
In order to pass these into the components, they have to be converted into JSON strings.
We also need to specify a few other parameters:

```python
expected_model_id = "my_diabetes_model:2"
target_column_name = "y"

treatment_feature_names = json.dumps(["bmi", "bp", "s2"])
desired_range = json.dumps([50, 120])

compute_name = "my_aml_compute_cluster"
```

With these in place, we can define the pipeline itself:
```python
from azure.ai.ml import dsl

@dsl.pipeline(
        compute=compute_name,
        description="Example RAI computation on diabetes decision making data",
        experiment_name=f"RAI_Diabetes_Decision_Example_RAIInsights_Computation_{model_name_suffix}",
    )
def rai_decision_pipeline(
        target_column_name,
        train_data,
        test_data,
        score_card_config_path
    ):
        # Fetch the model
        fetch_job = fetch_model_component(
            model_id=expected_model_id
        )
        
        # Initiate the RAIInsights
        create_rai_job = rai_constructor_component(
            title="RAI Dashboard Example",
            task_type="regression",
            model_info_path=fetch_job.outputs.model_info_output_path,
            train_dataset=train_data,
            test_dataset=test_data,
            target_column_name=target_column_name,
        )
        create_rai_job.set_limits(timeout=120)

        # Add causal analysis
        causal_job = rai_causal_component(
            rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
            treatment_features=treatment_feature_names
        )
        causal_job.set_limits(timeout=120)
        
        # Add counterfactual analysis
        counterfactual_job = rai_counterfactual_component(
            rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
            total_cfs=20,
            desired_range=desired_range,
        )
        counterfactual_job.set_limits(timeout=600)

        # Combine everything
        rai_gather_job = rai_gather_component(
            constructor=create_rai_job.outputs.rai_insights_dashboard,
            insight_2=causal_job.outputs.causal,
            insight_3=counterfactual_job.outputs.counterfactual,
        )
        rai_gather_job.set_limits(timeout=120)

        rai_gather_job.outputs.dashboard.mode = "upload"
        rai_gather_job.outputs.ux_json.mode = "upload"
        
        # Generate score card in pdf format for a summary report on model performance,
        # and observe distrbution of error between prediction vs ground truth.
        rai_scorecard_job = rai_scorecard_component(
            dashboard=rai_gather_job.outputs.dashboard,
            pdf_generation_config=score_card_config_path
        )

        return {
            "dashboard": rai_gather_job.outputs.dashboard,
            "ux_json": rai_gather_job.outputs.ux_json,
            "scorecard": rai_scorecard_job.outputs.scorecard
        }
```
This has defined a function which can create AzureML pipelines.
To create the pipeline itself, we call it:
```python
insights_pipeline_job = rai_decision_pipeline(
    target_column_name=target_feature,
    train_data=diabetes_train_pq,
    test_data=diabetes_test_pq,
    score_card_config_path=score_card_config_path
)
```

### Submitting the Pipeline

With a pipeline object created, we submit it using the `MLClient`:
```python
ml_client.jobs.create_or_update(insights_pipeline_job)
```
The job should be created in the AzureML portal, and we can watch its progress.
Once complete, if we navigate to the 'Model' page in the AzureML portal,
the analysis should have appeared in the 'Responsible AI (preview)' tab.