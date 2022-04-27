# Setup for Responsible AI
The goal of this document is to get your Responsible AI components registered and ready for use in your Azure Machine Learning Workspace

**Note:** The following instructions are for v2.1.2 of the SDK. There may be problems using later SDK releases due to changing APIs.

## Setup on local machine (Recommended)

### Pre-requisites

1. AzureML Workspace with a compute cluster. We strongly recommend using an existing test or sandbox Workspace or creating a new Workspace because the private preview bits can have bugs. DO NOT TRY THE PREVIEW ON A WORKSPACE WITH PRODUCTION ASSETS.
1. If you do not have the Azure CLI installed, [follow the installation instructions](https://docs.microsoft.com/cli/azure/install-azure-cli). The minimum version you'll need is 2.30. Check the version with `az version`. You can use [Azure Cloud Shell](https://docs.microsoft.com/en-us/azure/cloud-shell/quickstart) which has Azure CLI pre-installed.
1. Once the CLI is installed, [add the CLI v2 bits here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli)
1. Set your environment variables such as workspace, region, and subscription ID that you would like to work in.
    ```powershell
    az login
    az account set -s "<YOUR_SUBSCRIPTION_NAME_OR_ID>"
    az configure --defaults group="<your_resource_group_name>" location="<your_azure_region>" workspace="<your_workspace_name>"
    ```
1. (Optional) Familiarize yourself with [CLI 2.0 Jobs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-cli)

### RAI Private Package install

1. Create a local conda environment with Python 3.8
    ```
    conda create -n [env name] python=3.8
    conda activate [env name]
    ```
1. Clone the git repo 
    ```powershell
    git clone https://github.com/Azure/RAI-vNext-Preview
    cd RAI-vNext-Preview
    ```
1. In the repo root:

    a. Install the prerequisites

    ```powershell
    pip install -r requirements-dev-releasepackage.txt
    ```

    Verify that you have SDK version 2.1.2 by running:

    ```powershell
    pip show azure-ml
    ```
    If not, add `--upgrade` to the `pip install` command above.

    b. Generate the both configuration files by answering prompts from:

    ```powershell
    python ./scripts/generate_registration_files.py
    ```

    c. Register all the RAI components:

    ```powershell
    python scripts/register_azureml.py --workspace_config config.json --component_config component_config.json --base_directory .
    ```


1. Validate that your components have been registered in your workspace at https://ml.azure.com by going to the Components tab on the left hand workspace navigation list, and looking for entries like "Gather RAI Insights Dashboard". If your workspace does not have a Components tab, go to the Pipelines tab, and create a new pipeline. In the pipeline designer, you should be able to find the RAI components.

1. (Optional) Install the OSS version of the Responsible AI dashboard:

```powershell
pip install jupyter markupsafe<=2.0.1 itsdangerous==2.0.1
pip install responsibleai~=0.18.0 raiwidgets~=0.18.0 pyarrow
```

1. (Optional) Install the miniature SDK for the RAI components. This allows dashboards created in AzureML to be downloaded to your local machine.
    ``` powershell
    pip install -e src/azure-ml-rai
    ```
## Next Steps
- Build your first Model Analysis in the CLI or SDK

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
pip install responsibleai~=0.18.0 raiwidgets~=0.18.0 pyarrow
```
``` powershell
pip install -r requirements-dev-releasepackage.txt
```
Optionally, install our minature SDK
``` powershell
pip install -e src/azure-ml-rai
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

