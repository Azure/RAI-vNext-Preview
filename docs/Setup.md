# Setup for Responsible AI
The goal of this document is to get your Responsible AI components registered and ready for use in your Azure Machine Learning Workspace

## Setup through Cloud shell (Recommended)
1. Go to https://shell.azure.com
2. Select a subscription to create a storage account and Microsoft Azure Files share.
3. Select "Create storage"
4. The cloud shell should have AML CLI pre-installed. To ensure you have the latest cmdlets installed run ```az extension add -n ml -y``` 
5. Make sure your setup is working with any of the list commands: ``` az ml compute list ```, ``` az ml jobs list ```, ``` az ml data list ```
6. Run the following command to enable the private preview features in your environment
```powershell 
$Env:AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=$true
```
7. Setup the git repo 
```powershell
git clone https://github.com/Azure/AutoML-vNext-Preview
cd AutoML-vNext-Preview```
git checkout -b riedgar-ms/full-notebook 
```
8. Run the following pip installs
``` powershell
pip install jupyter responsibleai pyarrow pandas shap
pip install --extra-index-url https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2 azure-ml
pip install azureml-core azureml-mlflow
pip install -e src/azureml-ml-rai
```
9. Create a JSON file called component_config.json containing: '{version:1}' as follows...
```powershell
New-item -Path . -Name "component_config.json" -ItemType "file"
nano component_config.json
```
10. Add the text ```{version:1}``` to the file, save and hit enter
11. Got to portal.azure.com
12. Click on "Machine Learning"
![image](https://user-images.githubusercontent.com/53354089/145263293-46ad90f4-a624-4bce-ac6d-10e82fe30061.png)

13. Find the workspace you would like to use and click into the workspace

![image](https://user-images.githubusercontent.com/53354089/145263425-fd248292-217d-47a2-a89c-adeada367a08.png)

14. Click download config.json

![image](https://user-images.githubusercontent.com/53354089/145263548-d4373ac8-7ee5-45fd-89b3-e7d3d9bbd77e.png)

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



## Setup on local machine


## FAQ
## Cloud Shell
If you accidentally minimize your cloud shell, you can retrieve it by clicking here
![image](https://user-images.githubusercontent.com/53354089/145258468-2c5c5e02-03bb-4aa6-9961-67fa1a32af77.png)
