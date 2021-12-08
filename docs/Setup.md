# Setup for Responsible AI
The goal of this document is to get your Responsible AI components up and running in your workspace.

## Setup through Cloud shell (Recommended)
1. Go to https://shell.azure.com
2. Select a subscription to create a storage account and Microsoft Azure Files share.
3. Select "Create storage"
4. The cloud shell should have AML CLI pre-installed. To ensure you have the latest cmdlets installed run ```az extension add -n ml -y``` 
5. Make sure your setup is working with any of the list commands: ``` az ml compute list ```, ``` az ml jobs list ```, ``` az ml data list ```
6. Run ```$Env:AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=$true``` to enable the private preview features in your environment
7. ```git clone https://github.com/Azure/AutoML-vNext-Preview```
8. ```cd AutoML-vNext-Preview```
9. ``` git checkout -b riedgar-ms/full-notebook ```
10. 






## Setup on local machine


## FAQ
## Cloud Shell
If you accidentally minimize your cloud shell, you can retrieve it by clicking here
![image](https://user-images.githubusercontent.com/53354089/145258468-2c5c5e02-03bb-4aa6-9961-67fa1a32af77.png)
