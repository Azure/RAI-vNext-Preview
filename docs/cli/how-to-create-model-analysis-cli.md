# How to create a Model Analysis Job in AzureML CLI

This sample will walk you through creating a simple Model analysis dashboard with a single component attatched.

## Prequisites

- Install the ``` az ml ``` CLI and register your Responsible AI components via [these instructions](https://github.com/Azure/RAI-vNext-Preview/blob/main/docs/Setup.md). Make sure to write down the version number specified when you run the `generate_registration_files.py` script.

## A Pipeline YAML

An AzureML training pipeline can be specified using a YAML file.
The following is a simple pipeline which trains a model, registers it with AzureML, and then runs an RAI model analysis on it:

```yaml
name: Pipeline_RAI_Job
experiment_name: Pipeline_RAI_Experiment
type: pipeline

inputs:
  target_column_name: income
  my_training_data:
    dataset: azureml:Adult_Train_PQ:VERSION_REPLACEMENT_STRING
    mode: ro_mount
  my_test_data:
    dataset: azureml:Adult_Test_PQ:VERSION_REPLACEMENT_STRING
    mode: ro_mount

outputs:
  my_model_directory:
    mode: upload
  rai_insights_dashboard:
    mode: upload
  model_info:
    mode: upload

compute: azureml:<REPLACE-WITH-COMPUTE-CLUSTER>

settings:
  component_job:
    datastore: azureml:workspaceblobstore
    environment: azureml:AML-RAI-Environment:VERSION_REPLACEMENT_STRING

jobs:
  train-model-job:
    type: component_job
    component: azureml:TrainLogisticRegressionForRAI:VERSION_REPLACEMENT_STRING
    inputs:
      training_data: ${{inputs.my_training_data}}
      target_column_name: ${{inputs.target_column_name}}
    outputs:
      model_output: ${{outputs.my_model_directory}}

  register-model-job:
    type: component_job
    component: azureml:RegisterModel:VERSION_REPLACEMENT_STRING
    inputs:
      model_input_path: ${{jobs.train-model-job.outputs.model_output}}
      model_base_name: component_registered_lr_01
    outputs:
      model_info_output_path: ${{outputs.model_info}}

  create-rai-job:
    type: component_job
    component: azureml:RAIInsightsConstructor:VERSION_REPLACEMENT_STRING
    inputs:
      title: With just the OSS
      task_type: classification
      model_info_path: ${{jobs.register-model-job.outputs.model_info_output_path}}
      train_dataset: ${{inputs.my_training_data}}
      test_dataset: ${{inputs.my_test_data}}
      target_column_name: ${{inputs.target_column_name}}
      categorical_column_names: '["Race", "Sex", "Workclass", "Marital Status", "Country", "Occupation"]'
    outputs:
      rai_insights_dashboard: ${{outputs.rai_insights_dashboard}}

  explain_01:
    type: component_job
    component: azureml:RAIInsightsExplanation:VERSION_REPLACEMENT_STRING
    inputs:
      comment: Some random string
      rai_insights_dashboard: ${{jobs.create-rai-job.outputs.rai_insights_dashboard}}

  causal_01:
    type: component_job
    component: azureml:RAIInsightsCausal:VERSION_REPLACEMENT_STRING
    inputs:
      rai_insights_dashboard: ${{jobs.create-rai-job.outputs.rai_insights_dashboard}}
      treatment_features: '["Age", "Sex"]'
      heterogeneity_features: '["Marital Status"]'

  counterfactual_01:
    type: component_job
    component: azureml:RAIInsightsCounterfactual:VERSION_REPLACEMENT_STRING
    inputs:
      rai_insights_dashboard: ${{jobs.create-rai-job.outputs.rai_insights_dashboard}}
      total_CFs: 10
      desired_class: opposite

  error_analysis_01:
    type: component_job
    component: azureml:RAIInsightsErrorAnalysis:VERSION_REPLACEMENT_STRING
    inputs:
      rai_insights_dashboard: ${{jobs.create-rai-job.outputs.rai_insights_dashboard}}
      filter_features: '["Race", "Sex", "Workclass", "Marital Status", "Country", "Occupation"]'

  gather_01:
    type: component_job
    component: azureml:RAIInsightsGather:VERSION_REPLACEMENT_STRING
    inputs:
      constructor: ${{jobs.create-rai-job.outputs.rai_insights_dashboard}}
      insight_1: ${{jobs.causal_01.outputs.causal}}
      insight_2: ${{jobs.counterfactual_01.outputs.counterfactual}}
      insight_3: ${{jobs.error_analysis_01.outputs.error_analysis}}
      insight_4: ${{jobs.explain_01.outputs.explanation}}
```
Save this to a file (e.g. `rai-pipeline.yaml`) and update all of the instances of `VERSION_REPLACEMENT_STRING` with the version number you specified when running `generate_registration_files.py`.
You will also need to update the `compute:` line to point at the compute resource you wish to use. With those changes made, submit the job using
```bash
az ml job create --file ra-pipeline.yaml
```
The job should appear in AzureML studio, and you can watch its progress.
Once complete, go to the 'Models' view (in the left hand navigation bar) in AzureML studio in order to view your Responsible AI dashboard.
Search for the `component_registered_lr_01` model, and click into the the model details.
Select the 'Responsible AI (preview)' tab, and then click on the analysis you have just created.