#!/usr/bin/env python3

from azureml.core import Workspace


# this will create the machine learning workspace
ws = Workspace.create(name='myworkspace',
                      subscription_id='xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx',
                      resource_group='myrg',
                      create_resource_group=True,
                      location='canadacentral'
                     )

# this will write a config file which you can load
# it will write it in "<path>/.azureml/<file_name>"
ws.write_config(path="/Users/glenyu", file_name="ws_config.json")


# once you have your workspace config, you can load it for future work
# ws = Workspace.from_config(path="/Users/glenyu/.azureml/ws_config.json")
