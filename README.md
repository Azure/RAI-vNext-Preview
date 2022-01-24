# Azure Machine Learning Responsible AI Dashboard - Private Preview

Welcome to the private preview for the new Responsible AI dashboard in Azure Machine Learning (AzureML) SDK and studio. The following is a guide for you to onboard to the new capabilities. For questions, please contact mithigpe@microsoft.com.

## What is this new feature?

AzureML currently supports both [model explanations](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-machine-learning-interpretability-aml) and [model fairness](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-machine-learning-fairness-aml) in public preview. As we expand our offerings under Responsible AI tools for AzureML users, this new feature brings pre-existing features and brand new offerings under one-stop-shop SDK package and studio UI dashboard:
- Error Analysis (new): view and understand the error distributions of your model over your dataset via a decision tree map or heat map visualization.
- Data Explorer: explore your dataset by feature sets and other metrics such as predicted Y or true Y
- Model Statistics: explore the distribution of your model outcomes and performance metrics
- Interpretability: view the aggregate and individual feature importances across your model and dataset
- Counterfactual What-If's (new): create automatically generated diverse sets of counterfactual examples for each datapoint that is minimally perturbed in order to switch its predicted class or output. Also create your own counterfactual datapoint by perturbing feature values manually to observe the new outcome of your model prediction.
- Causal Analysis (new): view the aggregate and individual causal effects of *treatment features* (features which you are interested in controlling) on the outcome in order to make informed real-life business decisions. See recommended treatment policies for segmentations of your population for features in your dataset to see the effect on your real-life outcomes. 

This new feature offers users a new powerful and robust toolkit for understanding your model and data in order to develop your machine learning models responsibly, now all in one place and integrated with your AzureML workspace.

❗ **Please note:** This initial version of the Responsible AI dashboard currently does not support the integration of fairness metrics. For fairness metrics, please refer to our existing offering [here.](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-machine-learning-fairness-aml)

## Supported scenarios, models and datasets

`azureml-responsibleai` supports computation of Responsible AI insights for `scikit-learn` models that are trained on `pandas.DataFrame`. The `azureml-responsibleai` package accepts both models and SciKit-Learn pipelines as input as long as the model or pipeline implements a `predict` or `predict_proba` function that conforms to the `scikit-learn` convention. If not compatible, you can wrap your model's prediction function into a wrapper class that transforms the output into the format that is supported (`predict` or `predict_proba` of `scikit-learn`), and pass that wrapper class to modules in `azureml-responsibleai`.

Currently, we support datasets having numerical and categorical features. The following table provides the scenarios supported for each of the four responsible AI insights:-

| RAI insight | Binary classification | Multi-class classification | Multilabel classification | Regression | Timeseries forecasting | Categorical features | Text features | Image Features | Recommender Systems | Reinforcement Learning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | -- |
| Explainability | Yes | Yes | No | Yes | No | Yes | No | No | No | No |
| Error Analysis | Yes | Yes | No | Yes | No | Yes | No | No | No | No |
| Causal Analysis | Yes | No | No | Yes | No | Yes (max 5 features due to computational cost) | No | No | No | No |
| Counterfactual | Yes | Yes | No | Yes | No | Yes | No | No | No | No |

This is all available via Python SDK or CLI.

## Set Up
In this section, we will go over the basic setup steps that you need in order to generate Responsible AI insights for your models from SDK v2, CLI v2 and visualize the generated Responsible AI insights in [AML studio](https://ml.azure.com/).

### Create an AzureML workspace
Create an AzureML workspace by using the [configuration notebook](https://github.com/Azure/MachineLearningNotebooks/blob/master/configuration.ipynb)

### Install the required packages
In order to install `azureml-responsibleai` package you will need a python virtual environment. You can create a python virtual environment using `conda`.
```c
conda create -n azureml_env python=3.8
activate azureml_env
```

After activating your environment, if this is your first time running the RAI Dashboard in private preview then continue to the [setup instructions](https://github.com/Azure/RAI-vNext-Preview/blob/mergePrivatePreview_MAD/docs/Setup.md) to do a one-time setup for your workspace.



### Generating Responsibleai AI Dashboard insights
Once you have created an Azure workspace and registered your components in the one-time setup above, you can create a Responsible AI dashboard via the CLI or SDK. Start here for `examples` [folder](examples) to get started.

### Viewing your Responsible AI Dashboard in the AzureML studio portal
After generating the Responsible AI insights, you can view them in your associated workspace in AzureML studio, under your model registry.

![01](images/01_model_registry.png)
1. Go to your model registry in your AzureML studio workspace
2. Click on the model for which you've uploaded your Responsible AI insights

![02](images/02_model_details.png)
3. Click on the tab for `Responsible AI dashboard (preview)` under your model details page

![03](images/03_responsibleaitoolbox.png)
4. Under the `Responsible AI dashboard (preview)` tab of your model details, you will see a list of your uploaded Responsible AI insights. You can upload more than one Responsible AI dashboard for each model. Each row represents one dashboard, with information on which components were uploaded to each dashboard (i.e. explanations, counterfactuals, etc).

![04](images/04_dashboard.png)
5. At anytime while viewing the dashboard, if you wish to return to the model details page, click on `Back to model details`
<ol type="A">
  <li>You can view the dashboard insights for each component filtered down on a cohort you specify (or view all the data with the global cohort). Hovering over the cohort name will show the number of datapoints and filters in that cohort as a tooltip.</li>
  <li>Switch which cohort you are applying to the dashboard.</li>
  <li>Create a new cohort based on filters you can apply in a flyout panel.</li>
  <li>View a list of all cohorts created and duplicate, edit or delete them.</li>
  <li>View a list of all Responsible AI components you've uploaded to this dashboard as well as deleting components. The layout of the dashboard will reflect the order of the components in this list.</li>
</ol>

❗ **Please note:** Error Analysis, if generated, will always be at the top of the component list in your dashboard. Selecting on the nodes of the error tree or tiles of the error heatmap will automatically generate a temporary cohort that will be populated in the components below so that you can easily experiment with looking at insights for different areas of your error distribution.

![05](images/05_add_dashboard.png)
6. In between each component you can add components by clicking the blue circular button with a plus sign. This will pop up a tooltip that will give you an option of adding whichever Responsible AI component you enabled with your SDK.

#### Known limitations of viewing dashboard in AzureML studio
Due to the (current) lack of active compute, the dashboard in AzureML studio has fewer features than the dashboard generated with the open source package. To generate the full dashboard in a Jupyter python notebook, please download and use our [open source Responsible AI Dashboard SDK](https://github.com/microsoft/responsible-ai-widgets). 

Some limitations in AzureML studio include:
- Retraining of the Error analysis tree on different features is disabled
- Switching the Error analysis heat map to different features is disabled
- Viewing the Error analysis tree or heatmap on a smaller subset of your full dataset passed into the dashboard (requires retraining of the tree) is disabled
- ICE (Individual Conditional Expectation) plots in the feature importance tab for explanations are disabled
- Manually creating a What-If datapoint is disabled; you can only view the counterfactual examples already pre-generated by the SDK
- Causal analysis individual what-if is disabled; you can only view the individual causal effects of each individual datapoint

However, if you create a dashboard in AzureML, and then download it to a Jupyter notebook, it will be fully featured when running in that notebook.

## Responsible AI Dashboard walkthrough and sample notebooks
Please read through our [examples folder](examples) to see if this feature supports your use case. For more details about each individual component, please read through our brief [tour guide of the new Responsible AI dashboard capabilities.](https://github.com/microsoft/responsible-ai-widgets/blob/main/notebooks/responsibleaitoolbox-dashboard/tour.ipynb) 

## What Next?: How to join Private Preview 👀
We are super excited for you to try this new feature in AzureML! 
- Reach out to mithigpe@microsoft.com to enable your Azure subscription for this Private Preview feature.
- Fill out this form - Private Preview sign up for [Responsible AI Dashboard in AzureML](https://forms.office.com/r/R6PmBCkyWb)

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

