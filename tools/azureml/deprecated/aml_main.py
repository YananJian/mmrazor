import os
import sys
import logging
import errno
import subprocess as sp
import os.path as op
import argparse
import torch


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


def cmd_run(cmd, working_directory='./', succeed=False,
            return_output=False, output=None):

    if type(cmd) is str:
        cmd = cmd.strip().split(' ')

    e = os.environ.copy()
    e['PYTHONPATH'] = '/app/caffe/python:{}'.format(e.get('PYTHONPATH', ''))
    # in the maskrcnn, it will download the init model to TORCH_HOME. By
    # default, it is /root/.torch, which is different among diferent nodes.
    # However, the implementation assumes that folder is a share folder. Thus
    # only rank 0 do the data downloading. Here, we assume the output folder is
    # shared, which is the case in AML.
    e['TORCH_HOME'] = './output/torch_home'
    mkdir(e['TORCH_HOME'])
    
    logging.info('start to cmd run:\n{}\n'.format(' '.join(map(str, cmd))))

    if not return_output:
        try:
            if output:
                p = sp.Popen(cmd, stdin=sp.PIPE,
                            cwd=working_directory,
                            env=e, 
                            stdout=output,
                            stderr=output)
            else:
                p = sp.Popen(cmd, stdin=sp.PIPE,
                            cwd=working_directory,
                            env=e)
            p.communicate()
            if succeed:
                assert p.returncode == 0
        except:
            if succeed:
                logging.info('raising exception')
                raise
    else:
        return sp.check_output(cmd)


def init_logging():
    import socket
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d {} %(process)d %(filename)s:%(lineno)s %(funcName)10s(): %(message)s'.format(
            socket.gethostname()), datefmt='%m-%d %H:%M:%S')


def main():
    from pprint import pformat
    logging.info(pformat(sys.argv))

    parser = argparse.ArgumentParser(description='AML Experiment Test')
    parser.add_argument('--input_dir', required=True, type=str,
                        help='input dir with datasets and pretrained models'
                             'This must be a mountpoint in azure storage.')
    parser.add_argument('--output_dir', required=True, type=str,
                        help='output directory, a mountpoint in azure storage.')
    parser.add_argument('--config_file', required=True, type=str,
                        help='config file for job running')
    parser.add_argument('--dataset_dir', type=str, default='datasets',
                        help='dataset dir w.r.t. input_dir')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='pretrained model path w.r.t. input_dir')
    parser.add_argument('--num_gpus', type=int, default=-1,
                        help='number of gpus to use')
    parser.add_argument('--extra_args', type=str, default="", required=False,
                        help='extra arguments to modify the config file')
    args = parser.parse_args()

    data_dir = os.path.join(args.input_dir, args.dataset_dir)
    model_dir = os.path.join(args.input_dir, args.model_dir)

    # setup environment
    os.system("pip install -r requirements.txt")
    mkdir(args.output_dir)
   
    # copy code to /tmp/code
    working_dir = "/tmp/code"
    mkdir(working_dir)
    result = sp.run(['pwd'], stdout=sp.PIPE)
    root_dir = result.stdout.decode()
    root_dir = root_dir.strip()
    print("root_dir:{}".format(root_dir))
    os.system("cp -r . {}".format(working_dir))

    # create softlinks for datasets, models, and output
    cmd_run(['ln', '-s', data_dir, op.join(working_dir, 'datasets')])
    cmd_run(['ln', '-s', args.output_dir, op.join(working_dir, 'output')])
    cmd_run(['ln', '-s', model_dir, op.join(working_dir, 'models')])
    cmd_run('ls {}'.format(op.join(working_dir, 'models')))
    cmd_run('ls {}'.format(op.join(working_dir, 'output')))
    cmd_run('ls {}'.format(op.join(working_dir, 'datasets')))
    cmd_run('df -h')  # check shm size
 
    with open(op.join(working_dir, 'output','build_maskrcnn.log'), 'w') as output_f:
        cmd = 'python setup.py build develop'
        cmd_run(cmd, working_directory=working_dir, output=output_f)

    cmd_run('ls {}'.format(working_dir))
    with open(op.join(working_dir, 'output','working_dir_list.txt'), 'w') as output_f:
        cmd_run('ls {}'.format(working_dir), output=output_f)

    if args.num_gpus > 1:
        script = "python -m torch.distributed.launch --nproc_per_node={} " \
                "tools/train_net.py".format(args.num_gpus)
    elif args.num_gpus < 0:
        script = "python -m torch.distributed.launch --nproc_per_node={} " \
                "tools/train_net.py".format(torch.cuda.device_count())
    else:
        script = "python tools/train_net.py"

    cmd = "{} --config-file {}".format(script, args.config_file)
    if args.extra_args != "":
        cmd = cmd + " {}".format(args.extra_args)
    cmd_run(cmd, working_directory=working_dir)


if __name__ == "__main__":
    init_logging()
    main()
