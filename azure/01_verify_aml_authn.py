#!/usr/bin/env python3

import os
import json
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

SUBSCRIPTION_ID = os.environ.get('AZURE_SUBSCRIPTION_ID')
TENANT_ID       = os.environ.get('AZURE_TENANT_ID')
CLIENT_ID       = os.environ.get('AZUREML_CLIENT_ID')
CLIENT_SECRET   = os.environ.get('AZUREML_CLIENT_SECRET')


# Service Principal created following https://aka.ms/aml-notebook-auth
svc_pr = ServicePrincipalAuthentication(
  tenant_id=TENANT_ID,
  service_principal_id=CLIENT_ID,
  service_principal_password=CLIENT_SECRET
  )

ws = Workspace(
  subscription_id=SUBSCRIPTION_ID,
  workspace_name='ghive',
  resource_group='amlRG',
  auth=svc_pr
  )

print("Found workspace {} at location {}".format(ws.name, ws.location))
