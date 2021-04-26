# README
This guide will show you how to use Azure Machine Learning to create and train models

## Dependencies (pip)
- [`azureml-sdk`](https://pypi.org/project/azureml-sdk/)

## IMPORTANT!!! -- Setup
Before you you begin, I highly recommend first setting up an [Azure ML Workspace](https://docs.microsoft.com/en-us/azure/machine-learning/concept-workspace) by following the [script](./00_setup_aml_workspace.py)

Creating a workspace will create the following (in addition to the resource group):
- Workspace
- Key Vault
- Storage Account
- Application Insights

### Additional Resources
- [Authentication in Azure ML](aka.ms/aml-notebook-auth)
- [YouTube: Getting Started with Azure Machine Learning](https://www.youtube.com/watch?v=GBDSBInvz08)
