# How to create a Model Analysis Job in AzureML CLI

This sample will walk you through creating a simple Model analysis dashboard with a single component attatched.

## Prequisites

Ensure you have completed the [setup instructions](https://github.com/Azure/RAI-vNext-Preview#set-up)

## A Pipeline YAML

An AzureML training pipeline can be specified using a YAML file.
The following is a simple pipeline which trains a model, registers it with AzureML, and then runs an RAI model analysis on it:

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json
experiment_name: AML_RAI_Pipeline_from_YAML
type: pipeline

inputs:
  target_column_name: income
  my_training_data:
    type: uri_file
    path: azureml:adult_train_pq:1
    mode: download
  my_test_data:
    type: uri_file
    path: azureml:adult_test_pq:1
    mode: download

settings:
  default_datastore: azureml:workspaceblobstore
  default_compute: azureml:cpucluster
  continue_on_step_failure: false

jobs:
  train_model_job:
    type: command
    component: azureml:train_logistic_regression_for_rai:1
    inputs:
      training_data: ${{parent.inputs.my_training_data}}
      target_column_name: ${{parent.inputs.target_column_name}}


  register_model_job:
    type: command
    component: azureml:register_model:1
    inputs:
      model_input_path: ${{parent.jobs.train_model_job.outputs.model_output}}
      model_base_name: rai_cli_example_model

  create_rai_job:
    type: command
    component: azureml:rai_insights_constructor:1
    inputs:
      title: With just the OSS
      task_type: classification
      model_info_path: ${{parent.jobs.register_model_job.outputs.model_info_output_path}}
      train_dataset: ${{parent.inputs.my_training_data}}
      test_dataset: ${{parent.inputs.my_test_data}}
      target_column_name: ${{parent.inputs.target_column_name}}
      categorical_column_names: '["Race", "Sex", "Workclass", "Marital Status", "Country", "Occupation"]'

  explain_01:
    type: command
    component: azureml:rai_insights_explanation:1
    inputs:
      comment: Some random string
      rai_insights_dashboard: ${{parent.jobs.create_rai_job.outputs.rai_insights_dashboard}}

  causal_01:
    type: command
    component: azureml:rai_insights_causal:1
    inputs:
      rai_insights_dashboard: ${{parent.jobs.create_rai_job.outputs.rai_insights_dashboard}}
      treatment_features: '["Age", "Sex"]'
      heterogeneity_features: '["Marital Status"]'

  counterfactual_01:
    type: command
    component: azureml:rai_insights_counterfactual:1
    inputs:
      rai_insights_dashboard: ${{parent.jobs.create_rai_job.outputs.rai_insights_dashboard}}
      total_CFs: 10
      desired_class: opposite

  error_analysis_01:
    type: command
    component: azureml:rai_insights_erroranalysis:1
    inputs:
      rai_insights_dashboard: ${{parent.jobs.create_rai_job.outputs.rai_insights_dashboard}}
      filter_features: '["Race", "Sex", "Workclass", "Marital Status", "Country", "Occupation"]'

  gather_01:
    type: command
    component: azureml:rai_insights_gather:1
    inputs:
      constructor: ${{parent.jobs.create_rai_job.outputs.rai_insights_dashboard}}
      insight_1: ${{parent.jobs.causal_01.outputs.causal}}
      insight_2: ${{parent.jobs.counterfactual_01.outputs.counterfactual}}
      insight_3: ${{parent.jobs.error_analysis_01.outputs.error_analysis}}
      insight_4: ${{parent.jobs.explain_01.outputs.explanation}}
```
Save this to a file (e.g. `rai-pipeline.yaml`).
You will need to update the `default_compute:` line to point at the compute resource you wish to use.
With those changes made, submit the job using
```bash
az ml job create --file rai-pipeline.yaml
```
The job should appear in AzureML studio, and you can watch its progress.
Once complete, go to the 'Models' view (in the left hand navigation bar) in AzureML studio in order to view your Responsible AI dashboard.
Search for the `rai_cli_example_model` model, and click into the the model details.
Select the 'Responsible AI dashboard (preview)' tab, and then click on the analysis you have just created.