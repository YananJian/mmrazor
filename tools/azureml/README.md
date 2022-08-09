# AML Tools
Support cross-node distributed pytorch training on AML for `maskrcnn_benchmark` repo. 

Note the AML tools is improved from previous version. The old version is in `deprecated\`. 

## Setup
Assume project folder is the main folder of this repo.: `PROJECT_FOLDER = maskrcnn-benchmark/`.
1) Install AML python-sdk to use the python interface. 
```bash
pip install --upgrade azureml-sdk
```

2) Install azcopy to enable file transfer to/from azure storage account. Type `azcopy` to see if it is already install on the local machine. If not, please follow this page for details: https://docs.microsoft.com/en-us/previous-versions/azure/storage/storage-use-azcopy-linux. Note we only support azcopy version 8 or below. 

3) Create a folder inside `PROJECT_FOLDER` and named `.azureml`. 

4) Download workspace config file and save to `.azureml/config.json`.
It looks like this:
```bash
{
  "subscription_id": your_subscription_id,
  "resource_group": your_resource_group,
  "workspace_name": your_workspace_name
}
```

5) Setup storage account and save to `.azureml/datastore.json`.
It looks like this:
```bash
{
  "datastore_name": your_datastore_name, 
  "account_name": your_account_name,
  "container_name": your_container_name,
  "account_key": your_account_key
}
```

6) Copy `environment.json` in this folder and save to `.azureml/environment.json`.

7) Create a file named `.amlignore` in your `PROJECT_FOLDER` and specify paths/files that you want to be ignored when uploading your code for job training. For example, you can ignore your local datasets, models, etc. 

8) Create a folder named `output/` to store your experiment information. Each job should be a subfolder inside `output/`.

9) Setup compute target and save to `.azureml/datastore.json`.
It looks like this:
```bash
{
  "compute_target": "v100x8"
}
```


## Submit a Job
Here is an example:
```bash
    python tools/azureml/aml_submit.py \
        --workspace .azureml/config.json \ 
	--exp_name your_experiment_name (eg. your alias) \
	--environment .azureml/environment.json \
	--datastore .azureml/datastore.json \
	--compute_target your_compute_target_name \
	--input_dir your_input_data_path_in_azure_storage_account \
	--output_dir output/my_experiment \
	--cmd "your_full_command_to_run" \
	--num_nodes 1 
```
Note: 
1) `--input_dir` should store your dataset and pre-trained models. It should be structured like this:
```
data (input_dir)
|---- datasets/ (all datasets, each in one subfolder)
|-------- coco_caption/
|-------- openimages_v5c/
|---- models/
|-------- captioning/ (caption models)
|-------- R-50.pkl (od models)
```
Notes: if you change the datasets' folder name, you can modify the command in the file `maskrcnn_benchmark/aml_setup.sh` to adjust the names, which uses symbolic link for the mappinp. 

2) If you have a local dataset folder, do not name it as `data` or `datasets` because if you ignore this folder, you will also ignore the subfolders inside `maskrcnn_benchmark/` that will cause your program to crush. I suggest to name it as `datasets1` and ignore that in `.amlignore`.

3) `--output_dir` is a directory that will save your job information locally and trained models/logs on your storage account. Note that we will hyperlink the experiment folder on AML node to be `output/` so your program should default to save to `output/`. 

4) `--cmd` should be the full command you want to run without `python` as we will add that in `aml_main.py`. For example, `--cmd "tools/train_net.py --config-file your_config_file"`. Do not forgot the "". 

5) We only upload the folder `tools/azureml/` to the node. It will zip the code and upload to your job directory on Azure storage. This way you do not need to worry about the code file size limit by AML. 

6) `--exp_name` should prefix with your alias, e.g. `yourAlias-something`.


## Quey Job status, cancel/resubmit job, and download results 
After submitting a job, it will create a config file to store job information in `output/my_experiment/aml_job_config.json`.

1) set an alias in your `~/.bashrc`.
```bash
alias aml="python tools/azureml/aml_job.py "
```

2) Then, use the following commands. 
```bash
# check job status
aml status output/my_experiment
# cancel the job
aml abort output/my_experiment 
# resubmit the job (suppose you made some changes on the code and want to use the exact command to resubmit. If you job is still running, it will ask if you want to abort the job or not)
aml resubmit output/my_experiment 
# get the stdout logs from AzureML
aml logs output/my_experiment
# get the results from Azure storage
aml results output/my_experiment
# remove a job directory both local and on azure storage
aml remove output/my_experiment
```

3) More jobs at once, all or last jobs.  
This applies to all the commands (status, abort, resubmit, logs, results). 
```bash
# get the status of jobs whose directory starts with this pattern
aml status output/my_experiment_ 
# get the status of all jobs in `output/`
aml status all 
# get the status of last submitted job
aml status last
```

## Use it for other projects
If you wish to use the tools in other projects, modify `aml_setup.sh` if necessary. This file is used to setup the paths and environment on AML node to be the same as local.  

