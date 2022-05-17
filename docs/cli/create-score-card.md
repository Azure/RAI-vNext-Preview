# rai insight setup instruction
## checkout source code
```powershell
# Set working directory to local existing RAI-vNext-Preview repo.
# conda activate {your rai conda virtual environment}
# Make a copy of config.json before checking out branch. Backing up and restoring them in powershell is included below, you will need to adjust the command base on your shell.
# cp config.json $env:TEMP
# checkout branch
git checkout kicha/main/raiinsight
# cp $env:TEMP/config.json config.json
# cp $env:TEMP/component_config.json component_config.json
```
## re-register components
```powershell
# set a new version for component config. You will need to set version to be current highest + 1
echo '{"version": 2}' > component_config.json

python scripts/register_azureml.py --workspace_config config.json --component_config component_config.json --base_directory .
```
## Create new environment and submit sample job
```powershell
# Create a new environment
az ml environment create --file ./src/responsibleai/docker_env/rai_score_card_env.yaml
# submit sample job
az ml job create --file .\test\rai\pipeline_scorecard_full_boston.yaml
```

## run standalone insight score card generation
score card creation can also be run from existing rai insight that is already generated. But integration with UI retrieval may be impacted if dashboard artifacts is missing rai_insights.json, or raiinsight is generated from a different workspace. A sample standalone score card yaml can be found below
```yaml
$schema: https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json
experiment_name: AML_RAI_SCORECARD
type: pipeline

inputs:
  raidashboard:
    type: uri_folder
    path: boston_analyze_dashboard
    mode: download

compute: azureml:kicha

jobs:
  scorecard_01:
     type: command
     component: azureml:rai_score_card@latest
     inputs:
       dashboard: ${{parent.inputs.raidashboard}}
       pdf_generation_config:
         type: uri_file
         path: ./boston_analyse/pdf_gen.json
         mode: download
       predefined_cohorts_json:
         type: uri_file
         path: ./boston_analyse/cohorts.json
         mode: download
```
Notice that raidashboard in inputs can also be a registered dataset. Example given is uploading raiinsight folder from local.

## json configuration for cohorts and score card generation config
refer to documentation for modifying the configurations:
[cohorts](https://microsoft.sharepoint-df.com/:w:/t/AzureMLResponsibleAI/ETztxpUyd7VEh5UjY2G06cUBJbDhrKEQRx6ka6cGh0i5UQ?e=nfU4qS)
[scorecard config](https://microsoft.sharepoint-df.com/:w:/t/AzureMLResponsibleAI/EcRxssK8LCpAqphZJf8Mv8oBLKbvpGSTnkeyeVU3XZkMLQ?e=WViGHt)
Sample with above example:
cohorts.json
```json
[
  {
    "name": "High Tax",
    "cohort_filter_list": [
      {
        "method": "greater",
        "arg": [
          300
        ],
        "column": "TAX"
      }
    ]
  },
  {
    "name": "Low Tax",
    "cohort_filter_list": [
      {
        "method": "less",
        "arg": [
          300
        ],
        "column": "TAX"
      }
    ]
  }
]
```
pdf_gen.json
```json
{
  "Model": {
    "ModelName": "Boston Analyze",
    "ModelType": "Regression",
    "ModelSummary": "This is a regression model for boston analyze"
  },
  "Metrics": {
    "mean_absolute_error": {
      "threshold": "<=20"
    },
    "mean_squared_error": {}
  },
  "FeatureImportance": {
    "top_n": 6
  },
  "DataExplorer": {
    "features": [
      "TAX",
      "AGE",
      "CRIM",
      "DIS"
    ]
  },
  "Cohorts": [
    "High Tax",
    "Low Tax"
  ]
}
```

## Tips
Notice use of @latest tag for referencing aml assets. For example
```yaml
component: azureml:rai_score_card@latest
```
This will save you time by always referencing latest so that you don't need to manually edit version when there is an update