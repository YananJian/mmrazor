import os
import sys
import os.path as op
import errno
import json
import argparse

from azureml.core import Experiment
from azureml.train.estimator import Estimator
from workspace_utils import get_all_run_env, print_workspace_info
from misc import bcolors


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


def yn_choice(message, default='y'):
    choices = 'Y/n' if default.lower() in ('y', 'yes') else 'y/N'
    choice = input("%s (%s) " % (message, choices))
    values = ('y', 'yes', '') if choices == 'Y/n' else ('y', 'yes')
    return choice.strip().lower() in values


def parse_args():
    parser = argparse.ArgumentParser(description='AzureML Job Submission')
    # The following are AML job submission related arguments.
    parser.add_argument('--workspace', required=False, type=str, 
                        default='.azureml/config.json',
                        help='json config file with workspace info.')
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
    parser.add_argument('--input_dir', type=str, default='input',
                        help='input relative path w.r.t. datastore')
    parser.add_argument('--output_dir', type=str, default='output', 
                        help='output relative path w.r.t. datastore')
    parser.add_argument('--exp_name', type=str, default='test', 
                        help='output relative path w.r.t. datastore')
    parser.add_argument('--config_file', required=True, type=str, 
                        help='config file for job running')
    parser.add_argument('--dataset_dir', type=str, default='datasets',
                        help='dataset path w.r.t. input_dir')
    parser.add_argument('--model_dir', type=str, default='models', 
                        help='pretrained model file w.r.t. input_dir')
    parser.add_argument('--num_gpus', type=int, default=4,
                        help='number of gpus to use')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='change maskrcnn config using the command-line')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    Run this file from the project root directory for AzureML job submission.
    For example: 
    python tools/azureml/aml_submit.py --config_file train_oid.yaml 
           --exp_name xiyin1 --output_dir output/20190730_test/

    This will create a folder on local directory and save the run config
    and job information for checking the job status and downloading the results. 
    The process is similar to philly-tools.
 
    Please use your alias to name your experiment so all your jobs will be grouped
    together on the Azure portol. Experiments cannot be deleted once created. So 
    please do not create too many experiment names. For output_dir, it is used both 
    locally and on azure and will be used to sync files between local machine and 
    azure. It is recommended to name your job to start with a date so your job 
    directory are all sorted in order. 
    """
    args = parse_args()
    tags = {'output_dir': args.output_dir}
    if not args.opts:
        extra_args = ""
        vars(args).pop("opts")  # avoid save this to submit_config
    else:
        extra_args = " ".join(args.opts)
        args.opts = extra_args
        tags['extra_args'] = extra_args

    output_dir = args.output_dir
    if os.path.isdir(output_dir):
        msg = bcolors.WARNING + "The job directory exists, do you want " \
                "to abort the job and overwrite?" + bcolors.ENDC
        res = yn_choice(msg)
        if res:
            os.system("python tools/azureml/aml_job.py abort " + output_dir)
        else:
            print("Please use a different job directory and resubmit.")
            sys.exit()

    # setup workspace environment for running
    ws, ds, env, compute_target = get_all_run_env(args.workspace, args.datastore,
                                        args.environment, args.compute_target)
    print_workspace_info(ws)
    # mount the input/output folders
    input_folder = ds.path(args.input_dir).as_mount()
    output_folder = ds.path(output_dir).as_mount()
    
    script_params={
        '--input_dir': input_folder,
        '--output_dir': output_folder,
        '--config_file': args.config_file,
        '--dataset_dir': args.dataset_dir,
        '--model_dir': args.model_dir,
        '--num_gpus': args.num_gpus
    }
    if extra_args:
        script_params['--extra_args'] = extra_args
   
    est = Estimator(
            source_directory='.',
            compute_target=compute_target,
            environment_definition=env,
            entry_script='tools/azureml/aml_main.py',
            script_params=script_params
    )

    exp = Experiment(ws, args.exp_name)
    exp_run = exp.submit(est, tags=tags)
    exp_details = exp_run.get_details()

    # get the run number
    run_number = sum(1 for _ in exp.get_runs())
    exp_details['run_number'] = run_number
    print(bcolors.OKGREEN + "Submitted run number: {} on experiment: {}"
            .format(run_number, args.exp_name) + bcolors.ENDC)

    # save running information
    mkdir(output_dir)
    with open(os.path.join(output_dir, 'submit_config.json'), 'w') as f:
        json.dump(vars(args), f)

    with open(os.path.join(output_dir, 'experiment_config.json'), 'w') as f:
        json.dump(exp_details, f)


