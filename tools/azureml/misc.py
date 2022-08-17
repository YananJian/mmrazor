import json
import os.path as op


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[31m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_json_cfg(config_file):
    if not op.isfile(config_file):
        return 
    with open(config_file, 'r') as f:
        cfg = json.load(f)
    return cfg


def yn_choice(message, default='y'):
    choices = 'Y/n' if default.lower() in ('y', 'yes') else 'y/N'
    choice = input("%s (%s) " % (message, choices))
    values = ('y', 'yes', '') if choices == 'Y/n' else ('y', 'yes')
    return choice.strip().lower() in values


def ask_if_cancel_job(job_status, op_after_remove='overwrite'):
    msg = bcolors.WARNING + "Job is {}, do you want to abort the " \
            "job and {}?".format(job_status, op_after_remove) + bcolors.ENDC
    return yn_choice(msg)
 
