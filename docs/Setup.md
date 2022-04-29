# Setup for Responsible AI
The goal of this document is to get your Responsible AI components registered and ready for use in your Azure Machine Learning Workspace

**Note:** The APIs for the latest AzureML SDK are still in preview and subject to change.

## Prerequisites

In the following, we assume that you have access to an AzureML workspace.

## Setup through Cloud shell 
1. Go to https://shell.azure.com
2. Choose a `bash` shell
3. (first time) Select a subscription to create a storage account and Microsoft Azure Files share.
4. (first time) Select "Create storage"
5. The cloud shell should have AML CLI pre-installed. To ensure you have the latest cmdlets installed run ```az extension add -n ml -y``` 
6. Make sure your setup is working with any of the list commands: ``` az -h ```
7. Install [MiniConda](https://docs.conda.io/en/latest/miniconda.html)
8. Restart your shell, and make sure that the `conda` commands are available
9. Clone the git repo 
```powershell
git clone https://github.com/Azure/RAI-vNext-Preview
cd RAI-vNext-Preview
```
10. Run the install script
```bash
./quick-setup.bash <CONDA-ENV-NAME> <SUBSCRIPTION-ID> <RESOURCEGROUP-NAME> <WORKSPACE-NAME>
```
11. Validate that the setup worked by checking that the components are showing up under your "custom components" folder in ml.azure.com
![image](https://user-images.githubusercontent.com/53354089/145264202-12105d3b-9fd9-4234-96ee-ea9c22a4aaa3.png)
If your workspace does not have a Components tab, go to the Pipelines tab, and create a new pipeline. In the pipeline designer, you should be able to find the RAI components.

 

## FAQ
## Cloud Shell
If you accidentally minimize your cloud shell, you can retrieve it by clicking here
![image](https://user-images.githubusercontent.com/53354089/145258468-2c5c5e02-03bb-4aa6-9961-67fa1a32af77.png)

