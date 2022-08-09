import argparse
import os
import os.path as op
import sys
import re

from workspace_utils import get_workspace
from azureml.core import Experiment
from misc import bcolors, get_json_cfg
from azure_storage_io import StorageAccount


def main(func, output):
    # supported operations:
    func_set = ('abort', 'status', 'resubmit', 'logs', 'results', 'remove')
    assert func in func_set, 'unsupported operation'

    njobs = 0
    if op.isdir(output):
        if has_aml_job_cfg(output):
            njobs = 1
            operate_on_one_job(func, output)
        else:
            print('no aml_job_config.json file in {}'.format(output))
    elif output.lower() == 'all':
        for f in os.listdir('output/'):
            fname = op.join('output', f)
            if op.isdir(fname) and has_aml_job_cfg(fname):
                njobs += 1
                operate_on_one_job(func, fname)
    elif output.lower() == 'last':
        # find last submitted jobs based on config file timestamp
        time_stamps = []
        for f in os.listdir('output/'):
            fname = op.join('output', f)
            if op.isdir(fname) and has_aml_job_cfg(fname):
                time_stamps.append([fname, op.getmtime(op.join(fname, 'aml_job_config.json'))])
        if len(time_stamps) > 0:
            time_stamps = sorted(time_stamps, key = lambda x : x[1])
            operate_on_one_job(func, time_stamps[-1][0])
    else:
        for f in os.listdir('output/'):
            fname = op.join('output/', f)
            if op.isdir(fname) and re.search(output, fname) and \
               has_aml_job_cfg(fname):
                njobs += 1
                operate_on_one_job(func, fname)
    print("Processed {} jobs.".format(njobs))


def has_aml_job_cfg(output_dir):
    cfg_file = op.join(output_dir, 'aml_job_config.json')
    return op.isfile(cfg_file)


def get_aml_job_cfg(output_dir):
    cfg_file = op.join(output_dir, 'aml_job_config.json')
    return get_json_cfg(cfg_file)


def get_run_from_job_directory(output_dir):
    aml_job_cfg = get_aml_job_cfg(output_dir)
    if aml_job_cfg is not None:
        run_id = aml_job_cfg['exp_details']['runId']
        ws = get_workspace(aml_job_cfg['workspace'])
        return get_run_from_workspace(ws, run_id)


def get_run_from_workspace(ws, run_id):
    exp_name = run_id.split('_')[0]
    exp = Experiment(ws, exp_name)
    run = None
    for r in exp.get_runs():
        if r.id == run_id:
            run = r
            break
    return run


def print_run_cmd(cmd):
    print(cmd)
    os.system(cmd)


def operate_on_one_job(func, output_dir):
    aml_job_cfg = get_aml_job_cfg(output_dir)
    exp_details = aml_job_cfg['exp_details']
    del aml_job_cfg['exp_details']
    ws = get_workspace(aml_job_cfg['workspace'])
    run_id = exp_details['runId']
    run = get_run_from_workspace(ws, run_id)
    print(bcolors.OKBLUE + 'run_number: {}, run_id: {}, output_dir: {} '
        'workspace: {}'.format(exp_details['run_number'], run_id,
        output_dir, ws.name) + bcolors.ENDC)

    if func == 'abort':
        if run.status.lower() not in ('canceled', 'completed', 'failed'):
            run.cancel()
            print('Job ' + bcolors.WARNING + 'canceled' + bcolors.ENDC)
        else:
            print('Job status: {}, skip cancel.'.format(run.status))
    elif func == 'status':
        print('Job status: ' + bcolors.OKGREEN + run.status + bcolors.ENDC)
    elif func == 'resubmit':
        cmd = 'python tools/azureml/aml_submit.py '
        for key in aml_job_cfg:
            if key == 'cmd':
                cmd += '--' + key + ' \"' + str(aml_job_cfg[key]) + '\" '
            else:
                cmd += '--' + key + ' ' + str(aml_job_cfg[key]) + ' '
        print_run_cmd(cmd)
    elif func == 'logs':
        run.download_files(output_directory=output_dir)
        print('Downloaded log files to: {}'.format(output_dir))
        log_file = op.join(output_dir, 'azureml-logs/70_driver_log_0.txt')
        if op.isfile(log_file):
            os.system('tail ' + log_file)
    elif func == 'results':
        # please install azcopy version 8 or below to use this function.
        sa = StorageAccount(aml_job_cfg['datastore'])
        cmd = 'azcopy --source https://{}.blob.core.windows.net/{}/{} ' \
              '--destination {} --source-key {} --recursive --quiet'.format(
                sa.account_name, sa.container_name, output_dir,
                output_dir, sa.account_key)
        print_run_cmd(cmd)
    elif func == 'remove':
        # please install Azure CLI to use this function
        # try: "curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash" to install on linux
        # or check https://docs.microsoft.com/en-us/cli/azure/install-azure-cli-apt?view=azure-cli-latest
        can_remove = True
        if run.status.lower() not in ('canceled', 'completed', 'failed'):
            from misc import ask_if_cancel_job
            if ask_if_cancel_job(run.status, 'remove'):
                run.cancel()
                print('Job ' + bcolors.WARNING + 'canceled' + bcolors.ENDC)
            else:
                can_remove = False
                print("Job is {}, cannot remove directory".format(run.status))

        if can_remove:
            sa = StorageAccount(aml_job_cfg['datastore'])
            cmd = 'az storage remove --account-name {} --account-key {} --container-name {} ' \
                '-n {} -r'.format(sa.account_name, sa.account_key, sa.container_name, output_dir)
            print_run_cmd(cmd)
            print_run_cmd('rm -rf {}'.format(output_dir))


if __name__ == "__main__":
    """
    This file support to query job status, cancel/resubmit a job,
    or get the job logs, model results.
    It takes two system arguments: one operation name
    and one output_dir with job information.
    output_dir can be one job directory, or a pattern that can
    match to a list of jobs in output/.

    Example usages:
    1) set an alias in ~/.bashrc
    >> alias aml="python tools/azureml/aml_job.py "

    2) use the supported features:
    >> aml status output/20190731_test/ ===> check job status
    >> aml abort output/20190731_test/ ===> abort the job
    >> aml logs output/20190731_test/ ===> get job log
    >> aml results output/20190731_test/ ===> get job output like models
    >> aml resubmit output/20190731_test/ ===> resubmit a job
    >> aml remove output/20190731_test/ ===> remove job directory both local and on azure storage

    3) the second parameter could be a pattern to match job names:
    >> aml status output/20190731_ ===> check status for all jobs start with this

    4) you can also use "all" to do an operation to all jobs in output/:
    >> aml status all ===> get the status for all jobs
    >> aml abort all ===> cancel all jobs
    """

    parser = argparse.ArgumentParser(description='AML job monitor')
    parser.add_argument('func', type=str,
            help='support: status, abort, resubmit, logs, results, remove')
    parser.add_argument('output_dir', type=str,
            help='can be a job directory or a pattern, all or last')
    args = parser.parse_args()

    main(args.func, args.output_dir)

