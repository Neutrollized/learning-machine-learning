#!/usr/bin/env python3

import os
from azureml.core import Workspace

SUBSCRIPTION_ID = os.environ.get('AZURE_SUBSCRIPTION_ID')
LOCATION        = 'canadacentral'


# this will create the machine learning workspace
ws = Workspace.create(
  name='ghive',
  subscription_id=SUBSCRIPTION_ID,
  resource_group='amlRG',
  create_resource_group=True,
  location=LOCATION
  )

# this will write a config file which you can load
# it will write it in "<path>/.azureml/<file_name>"
ws.write_config(path="/Users/glenyu", file_name="ws_config.json")

# once you have your workspace config, you can load it for future work
# ws = Workspace.from_config(path="/Users/glenyu/.azureml/ws_config.json")
