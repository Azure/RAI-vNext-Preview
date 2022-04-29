# Setup for Responsible AI
The goal of this document is to get your Responsible AI components registered and ready for use in your Azure Machine Learning Workspace

**Note:** The APIs for the latest AzureML SDK are still in preview and subject to change.

## Prerequisites

In the following, we assume that you have access to an AzureML workspace.

## Setup through Cloud shell 
1. Go to https://shell.azure.com
2. (first time) Select a subscription to create a storage account and Microsoft Azure Files share.
3. (first time) Select "Create storage"
4. The cloud shell should have AML CLI pre-installed. To ensure you have the latest cmdlets installed run ```az extension add -n ml -y``` 
5. Make sure your setup is working with any of the list commands: ``` az -h ```
6. Run the following command to enable the private preview features in your environment
```powershell 
$Env:AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=$true
```
7. Setup the git repo 
```powershell
git clone https://github.com/Azure/RAI-vNext-Preview
cd RAI-vNext-Preview
```
8. Run the following pip installs
``` powershell
pip install jupyter markupsafe<=2.0.1 itsdangerous==2.0.1
pip install responsibleai~=0.18.0 raiwidgets~=0.18.0 pyarrow
```
``` powershell
pip install -r requirements-dev-releasepackage.txt
```
9. Generate the both configuration files by answering prompts from:

    ```powershell
    python ./scripts/generate_registration_files.py
    ```
    
10. Run the following to register all of the RAI components

    ```powershell
    python scripts/register_azureml.py --workspace_config config.json --component_config component_config.json --base_directory .
    ```

11. Validate that the setup worked by checking that the components are showing up under your "custom components" folder in ml.azure.com
![image](https://user-images.githubusercontent.com/53354089/145264202-12105d3b-9fd9-4234-96ee-ea9c22a4aaa3.png)
If your workspace does not have a Components tab, go to the Pipelines tab, and create a new pipeline. In the pipeline designer, you should be able to find the RAI components.





 

## FAQ
## Cloud Shell
If you accidentally minimize your cloud shell, you can retrieve it by clicking here
![image](https://user-images.githubusercontent.com/53354089/145258468-2c5c5e02-03bb-4aa6-9961-67fa1a32af77.png)

