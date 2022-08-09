import os
import sys
import logging
import errno
import subprocess as sp
import os.path as op
import argparse
import zipfile
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
        format='%(asctime)s.%(msecs)03d {} %(process)d %(filename)s:' \
                '%(lineno)s %(funcName)10s(): %(message)s'.format(
            socket.gethostname()), datefmt='%m-%d %H:%M:%S')


def setup_working_dir(zip_file, working_dir='/tmp/code'):
    mkdir(working_dir)
    os.chdir(working_dir)
    zip_ref = zipfile.ZipFile(zip_file, 'r')
    zip_ref.extractall(path=working_dir)


def main():
    parser = argparse.ArgumentParser(description='AML Experiment Test')
    parser.add_argument('--input_dir', required=True, type=str,
                        help='azure mount path of input dir with datasets/ and models/')
    parser.add_argument('--output_dir', required=True, type=str,
                        help='output directory, a mountpoint in azure storage.')
    parser.add_argument('--num_nodes', required=False, type=int, default=1,
                        help='number of nodes')
    parser.add_argument('--node_rank', required=False, type=int, default=0,
                        help='node rank for cross-node distributed training')
    parser.add_argument('--cmd', type=str, default=None, required=False,
                        help='cmd to run')
    args = parser.parse_args()

    # setup working directories
    working_dir = '/tmp/code'
    code_zip_file = op.join(args.output_dir, 'code.zip')
    setup_working_dir(code_zip_file)

    # setup environment
    os.environ["AML_JOB_INPUT_PATH"] = args.input_dir
    os.environ["AML_JOB_OUTPUT_PATH"] = args.output_dir
    if op.isfile('aml_setup.sh'):
        os.system('chmod +x aml_setup.sh')
        os.system('./aml_setup.sh')

    # distributed training
    num_gpus_per_node = torch.cuda.device_count()
    if args.num_nodes > 1:
        # cross node distributed training
        master_node_params = os.environ['AZ_BATCH_MASTER_NODE'].split(':')
        master_addr = master_node_params[0]
        master_port = master_node_params[1]
        cmd = " PYTHONPATH=\"$(dirname $0)\":$PYTHONPATH python -m torch.distributed.launch --nproc_per_node={} --nnodes {} --node_rank {} " \
              "--master_addr {} --master_port {} {}".format(num_gpus_per_node, args.num_nodes,
                      args.node_rank, master_addr, master_port, args.cmd)
    else:
        # single node distributed training
        cmd = "python -m torch.distributed.launch --nproc_per_node={} {}".format(
                num_gpus_per_node, args.cmd)

    cmd_run(cmd, working_directory=working_dir)

    # remove the zip file to save storage
    if args.node_rank == 0 and op.isfile(code_zip_file):
        os.remove(code_zip_file)


if __name__ == "__main__":
    init_logging()
    main()
