# How to create a Model Analysis Job in AzureML CLI

This sample will walk you through creating a simple Model analysis dashboard with a single component attatched.

## Prequisites

- Install the ``` az ml ``` CLI and register your Responsible AI components via [these instructions](https://github.com/Azure/RAI-vNext-Preview/blob/main/docs/Setup.md). You will need to remember your responses to the questions posed by the `generate_registration_files.py` script - specifically, the version number specified.

## A Pipeline YAML

An AzureML training pipeline can be specified using a YAML file.