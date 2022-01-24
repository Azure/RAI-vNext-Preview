# Setup for Responsible AI
The goal of this document is to get your Responsible AI components registered and ready for use in your Azure Machine Learning Workspace

## Setup on local machine (Recommended)

### Pre-requisites

1. AzureML Workspace with a compute cluster. We strongly recommend using an existing test or sandbox Workspace or creating a new Workspace because the private preview bits can have bugs. DO NOT TRY THE PREVIEW ON A WORKSPACE WITH PRODUCTION ASSETS.
1. If you do not have the Azure CLI installed, [follow the installation instructions](https://docs.microsoft.com/cli/azure/install-azure-cli). 2.15 is the minimum version you'll need. Check the version with `az version`. You can use [Azure Cloud Shell](https://docs.microsoft.com/en-us/azure/cloud-shell/quickstart) which has Azure CLI pre-installed.
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
    pip install -r requirements-dev.txt
    ```

    b. Generate the both configuration files by answering prompts from:

    ```powershell
    python ./scripts/generate_registration_files.py
    ```

    c. Register all the RAI components:

    ```powershell
    python scripts/register_azureml.py --workspace_config config.json --component_config component_config.json --base_directory src/responsibleai
    python scripts/register_azureml.py --workspace_config config.json --component_config component_config.json --base_directory test
    ```


1. Validate that your components have been registered in your workspace at https://ml.azure.com by going to the Components tab on the left hand workspace navigation list, and looking for entries like "Gather RAI Insights Dashboard"


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
pip install jupyter responsibleai pyarrow pandas shap
```
``` powershell
pip install --extra-index-url https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2 azure-ml
```
``` powershell
pip install azureml-core azureml-mlflow
```
``` powershell
pip install -e src/azure-ml-rai
```
9. Create a JSON file called component_config.json containing: '{version:1}' as follows...
```powershell
New-item -Path . -Name "component_config.json" -ItemType "file"
nano component_config.json
```
10. Add the text ```{version:1}``` to the file, save and hit enter
11. Go to portal.azure.com
12. Click on "Machine Learning"
![image](https://user-images.githubusercontent.com/53354089/145263293-46ad90f4-a624-4bce-ac6d-10e82fe30061.png)

13. Find the workspace you would like to use and click into the workspace

![image](https://user-images.githubusercontent.com/53354089/145263425-fd248292-217d-47a2-a89c-adeada367a08.png)

14. Click download config.json


15. Load this file into your Cloud shell by clicking "upload" and finding the config file in your downloads

![image](https://user-images.githubusercontent.com/53354089/145263695-12553cc9-f0ac-477b-89a3-3eba18f07cc6.png)

16. Move the file from the base directory into the root github directory (this code assumes your are still in the AutoML-vNext-Preview directory)

```powershell
Move-Item -Path ..\config.json -Destination ..\AutoML-vNext-Preview
```
17. Run

```powershell
scripts/Register-AzureML.ps1 src/responsibleai/registration_config.json

```
18. Validate that the setup worked by checking that the components are showing up under your "custom components" folder in ml.azure.com
![image](https://user-images.githubusercontent.com/53354089/145264202-12105d3b-9fd9-4234-96ee-ea9c22a4aaa3.png)





 

## FAQ
## Cloud Shell
If you accidentally minimize your cloud shell, you can retrieve it by clicking here
![image](https://user-images.githubusercontent.com/53354089/145258468-2c5c5e02-03bb-4aa6-9961-67fa1a32af77.png)

