import os
import os.path as op
import json

from azureml.core import Workspace, Datastore, Environment
from misc import get_json_cfg, bcolors


def get_workspace(config_file=None):
    if config_file is None:
        config_file = ".azureml/config.json"
    assert op.isfile(config_file), \
          "Config file does not exist {}".format(config_file)
    return Workspace.from_config(config_file)


def print_workspace_info(ws):
    meta_info = {}
    meta_info['datastores'] = list(ws.datastores.keys())
    meta_info['environments'] = list(ws.environments.keys())
    meta_info['experiments'] = list(ws.experiments.keys())
    meta_info['compute_targets'] = list(ws.compute_targets.keys())
    print(bcolors.OKGREEN + "Workspace info: " + bcolors.ENDC)
    print("datastores: {}".format(meta_info['datastores']))
    print("environments: {}".format(meta_info['environments']))
    print("experiments: {}".format(meta_info['experiments']))
    print("compute_target: {}".format(meta_info['compute_targets']))
    return meta_info


def get_all_run_env(
        workspace='.azureml/config.json', 
        datastore='.azureml/datastore.json',
        environment='.azureml/environment.json',
        target='.azureml/compute_target.json'
    ):
    ws = get_workspace(workspace)
    ds = get_datastore(ws, datastore)
    env = get_environment(ws, environment)
    compute_target = get_compute_target(ws, target)
    return ws, ds, env, compute_target


def get_datastore(ws, ds_or_file=None):
    # ds_or_file can be either the datastore_name to retrieve existing 
    # datastores in the workspace, or it could be a config file that 
    # specify the datastore information. By default this file is 
    # located in .azureml/datastore.json if not given.

    if ds_or_file in ws.datastores:
        return Datastore.get(ws, ds_or_file)
    
    if ds_or_file is None:
        config_file = '.azureml/datastore.json'
    else:
        assert op.isfile(ds_or_file), \
            "datastore_name or config file does not exist {}".format(ds_or_file)
        config_file = ds_or_file
    cfg = get_json_cfg(config_file)

    return Datastore.register_azure_blob_container(
            workspace=ws,
            datastore_name=cfg['datastore_name'],
            account_name=cfg['account_name'],
            container_name=cfg['container_name'],
            account_key=cfg['account_key']
        )
  

def get_environment(ws, env_or_file=None, register_if_new=True):
    # similar to the get_datastore function
    # env_or_file can be an existing environment name to retrieve 
    # from the workspace, or it could be a json config file 
    # to define environment. By default this file is located in 
    # .azureml/environment.json if not given

    if env_or_file in ws.environments:
        return Environment.get(ws, env_or_file)

    if env_or_file is None:
        config_file = '.azureml/environment.json'
    else:
        assert op.isfile(env_or_file),  "environment_name or config " \
            "file does not exist {}".format(env_or_file)
        config_file = env_or_file
    cfg = get_json_cfg(config_file)

    env = Environment(cfg.pop('environment_name'))
    for key in cfg:
        if cfg[key] in ("False", "True", "false", "true"):
            cmd = "env." + key + "=" + cfg[key]
        else:
            cmd = "env." + key + "='" + cfg[key] + "'"
        exec(cmd)

    if register_if_new:
        env.register(ws)
    return env


def get_compute_target(ws, target=None):
    if target is None:
        config_file = '.azureml/compute_target.json'
        if op.isfile(config_file):
            cfg = get_json_cfg(config_file)
            target_name = cfg['compute_target']
        else:
            # by default, use the first one in the workspace.
            assert len(ws.compute_targets.keys()) > 0, \
                "no compute_targets availabel in workspace"
            target_name = list(ws.compute_targets.keys())[0]
    else:
        if op.isfile(target):
            cfg = get_json_cfg(target)
            target_name = cfg['compute_target']
        else:
            target_name = target
    return ws.compute_targets[target_name]

