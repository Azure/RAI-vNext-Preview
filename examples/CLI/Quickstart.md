# Quickstart - AzureML CLI Model Analysis

## Prequisites
To run the CLI and create a model analysis, make sure that you have completed the setup instructions per the below link.

-  [Setup instructions](https://github.com/Azure/RAI-vNext-Preview#set-up)

After following the setup instructions above, your AzureML workspace should have all the required RAI and training components in your workspace to train a simple model and run a model analysis on it. Next, we want to create a pipeline that will execute these components in order to create a model and model analysis.
We have a sample pipeline.yml file ready for you to run and view a model analysis in your workspace via the instructions below.
1. In `test/rai/pipeline_boston_analyse.yaml, replace all` `"VERSION_REPLACEMENT_STRING"` with the value "1" (unless you customised the setup script to upload a different component version). Additionally replace the `compute: azureml:cpucluster` with the compute you would like to use.
2. Run Pipeline CLI Command for Boston housing example
``` Powershell 
az ml job create --file /test/rai/pipeline_boston_analyse.yaml
```
3. View the Pipeline run in ml.azure.com in the experiments tab.

## Resources
- [az ml cli v2 documentation](https://docs.microsoft.com/en-us/cli/azure/ml?view=azure-cli-latest)
### Quick reference commands
Login from CLI:
```CLI
az login --tenant <your_tenant_name_such_us_microsoft.onmicrosoft.com>
```
Check defaults set:
```CLI
az configure
```
Set by default Resource Group:
```CLI
az configure --defaults group=<your_resource_group_name> location=<your_azure_region>
```
Set by default AML Workspace:
```CLI
az configure --defaults workspace=<your_workspace_name>
```




