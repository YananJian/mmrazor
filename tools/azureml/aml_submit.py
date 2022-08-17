import os
import sys
import os.path as op
import errno
import json
import argparse

from azureml.core import Experiment
from azureml.train.dnn import PyTorch, Nccl
from workspace_utils import get_workspace, get_all_run_env, print_workspace_info
from misc import bcolors, ask_if_cancel_job
from azure_storage_io import zip_code_and_upload


def mkdir(path):
    # if it is the current folder, skip.
    # otherwise the original code will raise FileNotFoundError
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def check_job_directory(output_dir):
    if os.path.isdir(args.output_dir):
        # check if job is still active (running or queued)
        from aml_job import get_run_from_job_directory
        run = get_run_from_job_directory(output_dir)
        if run is not None:
            if run.status.lower() not in ('canceled', 'completed', 'failed'):
                if ask_if_cancel_job(run.status):
                    run.cancel()
                    print("Job " + bcolors.WARNING + 'canceled' + bcolors.ENDC)
                else:
                    print("Please use a different job directory and resubmit.")
                    sys.exit()
        else:
            print("Directory exist but did not find active job, continue.")
    else:
        mkdir(args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description='AzureML Job Submission')
    # The following are AML job submission related arguments.
    parser.add_argument('--workspace', required=False, type=str, 
                        default='.azureml/config.json',
                        help='json config file with workspace info.')
    parser.add_argument('--exp_name', type=str, required=True, 
                        help='experiment name to hold all runs')
    parser.add_argument('--environment', required=False, type=str,
                        default='.azureml/environment.json',
                        help='json config file or the environment name in workspace')
    parser.add_argument('--datastore', required=False, type=str,
                        default='.azureml/datastore.json',
                        help='json config file or the datastore name in workspace')
    parser.add_argument('--compute_target', required=False, type=str,
                        default='.azureml/compute_target.json',
                        help='json config file or the target name')
    # The following are job running related arguments used in the entry file. 
    parser.add_argument('--input_dir', type=str, default='data/',
                        help='input relative path w.r.t. datastore')
    parser.add_argument('--output_dir', type=str, default='output/', 
                        help='output relative path w.r.t. datastore')
    parser.add_argument('--cmd', required=True, type=str, 
                        help='full cmd to run on each gpu')
    parser.add_argument('--num_nodes', type=int, default=1,
                        help='number of nodes, > 1 for cross-node distributed training.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    tags = {'output_dir': args.output_dir}

    check_job_directory(args.output_dir)

    # setup workspace environment for running
    ws, ds, env, compute_target = get_all_run_env(args.workspace, args.datastore,
                                        args.environment, args.compute_target)
    # print_workspace_info(ws)

    # mount the input/output folders
    input_dir = ds.path(args.input_dir).as_mount()
    output_dir = ds.path(args.output_dir).as_mount()

    # zip code and upload
    zip_code_and_upload(zip_filename=op.join(args.output_dir, 'code.zip'),
            datastore_cfg=args.datastore)

    script_params={
        '--input_dir': input_dir,
        '--output_dir': output_dir,
        '--cmd': args.cmd,
        '--num_nodes': args.num_nodes,
        # '--node_rank': '$AZ_BATCHAI_TASK_INDEX',
    }
   
    est = PyTorch(
        source_directory='tools/azureml/',
        compute_target=compute_target,
        environment_definition=env,
        entry_script='aml_main.py',
        script_params=script_params,
        node_count=args.num_nodes,
        distributed_training=Nccl()
    )

    exp = Experiment(ws, args.exp_name)
    exp_run = exp.submit(est, tags=tags)
    exp_details = exp_run.get_details()

    # get the run number
    run_number = sum(1 for _ in exp.get_runs())
    exp_details['run_number'] = run_number
    print(bcolors.OKGREEN + "Submitted run number: {} for experiment: {} " \
        "on workspace: {}".format(run_number, args.exp_name, ws.name) \
        + bcolors.ENDC)

    # save running information for job operations later on
    aml_job_config = vars(args)
    del exp_details['inputDatasets'] # delete arguments that cannot be dumped
    aml_job_config['exp_details'] = exp_details
    with open(op.join(args.output_dir, 'aml_job_config.json'), 'w') as f:
        json.dump(aml_job_config, f)

