# Quickstart - Azure ML CLI Responsible AI Insights

## Prequisites
To run the CLI and create Responsible AI Insights, make sure that you have completed the setup instructions per the below link.

-  [Setup instructions](https://github.com/Azure/RAI-vNext-Preview#set-up)

After following the setup instructions above, your AzureML workspace should have all the required RAI and training components in your workspace to train a simple model and run Responsible AI Analysis on it. Next, we want to create a pipeline that will execute these components in order to create a model and corresponding Responsible AI Insights.

In this directory there is a `pipeline_rai_adult.yaml` file, which runs a simple analysis on the well known 'Adult Census' dataset.
Assuming that you have completed all of the setup instructions, you can submit this to AzureML by running
```powershell
az ml job create --file pipeline_rai_adult.yaml
```
In the AzureML portal, you should see a new experiment created with the name 'RAI_CLI_Submission_Adult_1'.
Once this completes, you should see a model called 'rai_adult_cli_01' in the Models tab of the AzureML portal.
This will have the analysis available in the 'Responsible AI (preview)' tab.