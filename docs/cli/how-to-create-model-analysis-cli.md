# How to create a Model Analysis Job in AzureML CLI

This sample will walk you through creating a simple Model analysis dashboard with a single component attatched.
## Prequisites
- Install the ``` az ml ``` CLI via [these instructions](https://github.com/Azure/AutoML-vNext-Preview/blob/main/docs/cli/cli-installation.rst)


## Create a Model Analysis Pipeline
### Step 1: Download and register Model Analysis components for Private Preview
To install the private preview components for RAI model analysis first [download this zip file and save it in a retrievable location]().

Then, register your 'InitiateModelAnalysis.YAML' component and your 'GenerateExplanations' components from that directory. 
```YAML
az ml component create –file <InitiateModelAnalysis.yaml>
az ml component create –file <GenerateExplanations.yaml>
```

### Step 2: Provide inputs to your Initiate Model Analysis Component
- After downloading and registering your model analysis components, you will want to navigate to the folder that contains your components.
- Open the "Initiate_Model_Analysis.YAML" file


The .YAML config below will initiate your model analysis. There are a few parameters that will need to be ready in advance.

```YAML
$schema: http://azureml/sdk-2-0/CommandComponent.json
name: InitiateModelAnalysis
display_name: Model Analysis with azureml-responsibleai
version: VERSION_REPLACEMENT_STRING
type: command_component
inputs:
  title:
    type: string
  task_type:
    type: string # [classification, regression]
    enum: ['classification', 'regression']
  model_info_path:
    type: path # To model_info.json
  train_dataset:
    type: path # Must be Parquet
  test_dataset:
    type: path # Must be Parquet
  target_column_name:
    type: string
  X_column_names:
    type: string # List[str]
  datastore_name:
    type: string
  maximum_rows_for_test_dataset:
    type: integer
    default: 5000
  categorical_column_names:
    type: string # Optional[List[str]]
outputs:
  model_analysis_info:
    type: path
code:
  local_path: ./rai_analyse/
environment: azureml:AML-RAI-Environment:VERSION_REPLACEMENT_STRING
command: >-
  python create_model_analysis.py
  --title '${{inputs.title}}'
  --task_type ${{inputs.task_type}}
  --model_info_path ${{inputs.model_info_path}}
  --train_dataset ${{inputs.train_dataset}}
  --test_dataset ${{inputs.test_dataset}}
  --target_column_name ${{inputs.target_column_name}}
  --X_column_names '${{inputs.X_column_names}}'
  --datastore_name ${{inputs.datastore_name}}
  --maximum_rows_for_test_dataset ${{inputs.maximum_rows_for_test_dataset}}
  --categorical_column_names '${{inputs.categorical_column_names}}'
  --output_path ${{outputs.model_analysis_info}}
```

### Step 3: Provide inputs to your Model Explanation component
Once your initiate Model Analysis component is set, you will be able to add any compatible component to your analysis which will be rendered for you in a single dashboard. Each of these components have their own .YAML configurations, for this example we will be using the explanation component. [Explore other components here]()

``` YAML
$schema: http://azureml/sdk-2-0/CommandComponent.json
name: AzureMLModelAnalysisExplanation
display_name: Explanation with azureml-responsibleai
version: VERSION_REPLACEMENT_STRING
type: command

inputs:
  comment:
    type: string
  model_analysis_info:
    type: path

code:
  local_path: ./rai_analyse/

environment: azureml:AML-RAI-Environment:VERSION_REPLACEMENT_STRING

command: >-
  python create_explanation.py
  --comment '${{inputs.comment}}'
  --model_analysis_info ${{inputs.model_analysis_info}}

```

### Step 4: construct a pipeline job with your Model Analysis Components

### Step 5: submit your job through az ml CLI
```
az ml job create –file pipeline_model_analysis.yaml
```

You can also specify the Job's name as a CLI parameter so it'll override any job name specified in the .YAML config file, so you don't need to change the .YAML every time you create another run with the same .YAML (Since each Job's name need to be unique and cannot be repeated)
```
az ml job create --file AzureMLModelAnalysis.yaml --name <my-specific-job-name-02>
```

## Useful CLI Commands

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


