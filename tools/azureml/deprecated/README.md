# AzureML Concepts
Please install azureml-sdk package to use the python interface. 
The following codes are already implemented in different files under this folder. 
You just need to prepare the config files accordingly and the code should work fine. 
But I am providing more details on each concept for you to better understand how it works. 

## Workspace
Workspace is a place to hold all resources including compute targets (GPUs), datastores, environments, and experiments. 
To access a workspace, you need to have the information of `subscription_id`, `resource_group`, and `workspace_name`. 
Create a folder named `.azureml` in your project directory and save a config file named `config.json`.
It looks like this:
```bash
{
  "subscription_id": your_subscription_id,
  "resource_group": your_resource_group,
  "workspace_name": your_workspace_name
}
```
To create a workspace object, use the following command:
```bash
from azureml.core import Workspace
ws = Workspace.from_config()
```
By default it will read the config in `.azureml/config.json`. You could also provide a different config file name.
```bash
ws = Workspace.from_config(your_config_file)
```
If you encounter azure login issue in this step, try `az login --use-device-code` on your Linux terminal. 
It will direct you to a website for authentication. This step just needs to be done once. 

## Compute Target
It means the GPU compute resources, which is saved in the workspace. 
To check the compute target you have:
```bash
# get the list of compute target names 
ws.compute_targets.keys() 
# get the compute_target object
target = ws.compute_targets[target_name]
```
You can also store your compute target in a config file named `.azureml/compute_target.json`, which will be loaded during job submission. 
```bash
{
    "compute_target": your_compute_target_name
}
```

## Datastore
Datastore contains the Azure storage account information. It tells the job where to read the data and where to save the outputs. 
Datastore just need to be registered once and is available in the workspace to retrieve using datastore name. 
Create a file in `.azureml/datastore.json` that looks like this:
```bash
{
  "datastore_name": your_datastore_name, 
  "account_name": your_account_name,
  "container_name": your_container_name,
  "account_key": your_account_key
}
```
For the first time, register the datastore to the workspace.
```bash
from azureml.core import Datastore
ds = Datastore.regiter_azure_blob_container(
        workspace=ws,
        datastore_name=your_datastore_name,
        account_name=your_account_name,
        container_name=your_container_name,
        account_key=your_account_key
    )
```
Next time, you can retrieve your datastore directly:
```bash
ds = Datastore.get(ws, your_datastore_name)
```
A path on Azure storage should be converted to a mount point. 
```bash
# the input_folder
input_folder = ds.path(your_path_on_azure_storage).as_mount()
```

## Environment
Environment includes the running environment variables like docker image, python path, etc. 
Similar to datastore, environment can be registered to the workspace and be retrieved later. 
I have already registered an environment to run maskrcnn-benchmark repo. 
Please check the example of `environment.json` in this folder to see the settings. 
```bash
# to get a list of environment names
ws.environments.keys()
# to get an environment object
env = Environment.get(ws, "maskrcnn")
# to register an environment to a workspace
env.register(ws)
```

## Experiment
Experiment object holds a list of runs. 
One run is one job. 
You can consider it as a folder/place to organize your jobs. 
The experiment name cannot be removed from the workspace, so we recommend to use your alias as your experiment name and run all your jobs under this experiment. Experiment object can submit jobs and keep track of all you jobs. 
```bash
from azureml.core import Experiment
# to get the list of experiments in the workspace
ws.experiments.keys()
# to create and access the experiment object
exp = Experiment(ws, your_exp_name)
```


## Estimator
This object holds the details of the job you want to run. 
Check the example in `aml_submit.py`:
```bash
est = Estimator(
    source_directory='.', # the directory to upload the code
    compute_target=taget, # compute target object
    environment_definition=env, # environment object
    entry_script='tools/azureml/aml_main.py', # the entry script for this repo. 
    script_params={
       "--config_file": your_config_file
       }  # script_params are the parameters for your entry_script. 
    )
# submit the job
run = exp.submit(est)
```
This will upload the code in the source directory, setup the running environment, require GPU resources, and then run the entry script using the script parameters. 
Note that AML has a limit on the file size you can upload for each job. You can use `.amlignore` to ignore folders/files that you do not want to upload.
The format is similar to `.gitignore`. But it will ignore all folders with the pattern you specify, so make sure you do not accidently ignore a source code folder. 

## Run
This object holds the job details and provide function to check job status, cancel job, etc. 
```bash
# to get job detail
run.get_details()
# get all runs of one experiment
exp.get_runs()
```
After job submission, I will save the job details into the output directory and then we can retrieve the run object from the experiment and perform operations. 
These features are implemented in `aml_job.py`. 
```bash
# cancel job
run.cancel()
# check job status
run.status()
# download azureml logs
run.download_files(output_directory=output_dir)
```

# Use AML Tools for maskrcnn-benchmark repo.

**1. Submit a job using config file without modification**
```bash
python tools/azureml/aml_submit.py --config_file your_config_file.yaml --output_dir output/20190802_test/ --exp_name your_alias
```

**2. Submit a job with command modification**

You can also modify the config parameters using command line. But these parameters should be added last. 
```bash
python tools/azureml/aml_submit.py --config_file your_config_file.yaml --output_dir output/20190802_test/ --exp_name your_alias SOLVER.IMS_PER_BATCH 8 
```
After submitting a job, it will create a local folder to store the job details for job monitor. 

**3. Check job status, cancel job, and download results**

First, set an alias in your `~/.bashrc`.
```bash
alias aml="python tools/azureml/aml_job.py "
```

Then, use the following commands. 
```bash
# check job status
aml status output/20190802_test/
# cancel the job
aml abort output/20190802_test/ 
# resubmit the job
aml resubmit output/20190802_test/
# get the stdout logs from AzureML
aml logs output/20190802_test/
# get the results from Azure storage
aml results output/20190802_test/ 
```

You could also keep track of multiple jobs at once. 
This applies to all the commands (status, cancel, resubmit, logs, results). 
```bash
# get the status of jobs whose directory starts with this pattern
aml status output/20190802 
# get the status of jobs whose directory matches to this pattern
aml status "output/20190802_*_test*"
# get the status of all jobs in `output/`
aml status all 
```

# Use AML Tools in your own project. 
The files in this folder do not have dependency on the maskrcnn repo. 
You just need to modify the commands in `aml_main.py` to tell what to run, and what script parameters you want to pass to the main file. 
Then you should be able to use these features normally. 


