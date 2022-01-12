# Boston housing RAI Dashboard

## Prequisites
To run the CLI and create a model analysis, make sure that you have completed the setup instructions per the below link.

-  [Setup instructions](https://github.com/Azure/RAI-vNext-Preview/blob/main/docs/Setup.md)

After following the setup instructions above, your AzureML workspace should have all the required RAI and training components in your workspace to train a simple model and run a model analysis on it. Next, we want to create a pipeline that will execute these components in order to create a model and model analysis.
We have a sample pipeline.yml file ready for you to run and view a model analysis in your workspace via the instructions below.
1. In `pipeline.yaml` file, replace all `"VERSION_REPLACEMENT_STRING"` with the version number of the component you would like to use in your workspace. (If copying from first-time setup instructions, this should be version 1). 
2. Replace the `compute: azureml:cpucluster` with the compute you would like to use in your workspace.
3. Run Pipeline CLI Command for Boston housing example
``` Powershell 
az ml job create --file pipeline.yaml
```
3. View the Pipeline run in ml.azure.com in the experiments tab.
