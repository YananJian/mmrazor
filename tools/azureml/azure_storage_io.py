import os
import json
import zipfile

from ignore_file import AmlIgnoreFile
from misc import bcolors


class StorageAccount(object):
    def __init__(self, cfg_file='.azureml/datastore.json'):
        self.cfg_file = cfg_file
        with open(cfg_file, 'r') as f:
            cfg = json.load(f)
        self.datastore_name = cfg['datastore_name']
        self.account_name = cfg['account_name']
        self.container_name = cfg['container_name']
        self.account_key = cfg['account_key']


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


# make_zipfile_exclude function is adopted from azureml-sdk
def make_zipfile_exclude(base_dir, zip_filename, exclude_function):
    with zipfile.ZipFile(zip_filename, "w") as zf:
        for dirpath, dirnames, filenames in os.walk(base_dir):
            relative_dirpath = os.path.relpath(dirpath, base_dir)
            for name in sorted(dirnames):
                full_path = os.path.normpath(os.path.join(dirpath, name))
                relative_path = os.path.normpath(os.path.join(relative_dirpath, name))
                if not exclude_function(full_path):
                    zf.write(full_path, relative_path)
            for name in filenames:
                full_path = os.path.normpath(os.path.join(dirpath, name))
                relative_path = os.path.normpath(os.path.join(relative_dirpath, name))
                if not exclude_function(full_path):
                    if os.path.isfile(full_path):
                        zf.write(full_path, relative_path)


def zip_code_and_upload(project_dir='.', zip_filename='code.zip',
        datastore_cfg='.azureml/datastore.json', remove_zip_file=True):
    print(bcolors.OKGREEN + "Zip code and upload: " + bcolors.ENDC)
    ignore = AmlIgnoreFile(project_dir)
    make_zipfile_exclude(project_dir, zip_filename, ignore.is_file_excluded)
    upload_files_to_azure(zip_filename, datastore_cfg=datastore_cfg)
    if remove_zip_file:
        os.remove(zip_filename)


def upload_files_to_azure(local_file_or_path, azure_file_or_path=None,
        datastore_cfg='.azureml/datastore.json', overwrite_if_exist=True,
        verbose=False):
    if not azure_file_or_path:
        # assume the same file/folder structure
        azure_file_or_path = local_file_or_path
    storage_account = StorageAccount(datastore_cfg)
    if os.path.isfile(local_file_or_path):
        cmd = "azcopy --source {} --destination https://{}.blob.core.windows.net/" \
              "{}/{} --dest-key {} "
    elif os.path.isdir(local_file_or_path):
        cmd = "azcopy  --source {} --destination https://{}.blob.core.windows.net/" \
              "{}/{} --dest-key {} --recursive "
    else:
        raise ValueError("File or path cannot be found: {}".format(local_file_or_path))
    if overwrite_if_exist:
        cmd += "--quiet "
    if verbose:
        cmd += "--verbose "
    cmd = cmd.format(local_file_or_path, storage_account.account_name,
          storage_account.container_name, azure_file_or_path,
          storage_account.account_key)
    print(cmd)
    os.system(cmd)


def download_files_from_azure(azure_file_or_path, local_file_or_path=None,
        datastore_cfg='.azureml/datastore.json', overwrite_if_exist=True,
        verbose=False):
    if not local_file_or_path:
        local_file_or_path = azure_file_or_path
    if os.path.splitext(local_file_or_path)[1]:
        # for a file
        if not os.path.isdir(os.path.dirname(local_file_or_path)):
            print("create directory {}".format(os.path.dirname(local_file_or_path)))
            mkdir(os.path.dirname(local_file_or_path))
    else:
        # for a path
        if not os.path.isdir(local_file_or_path):
            print("create directory {}".format(local_file_or_path))
            mkdir(local_file_or_path)

    storage_account = StorageAccount(datastore_cfg)
    cmd = "azcopy --source https://{}.blob.core.windows.net/{}/{} --destination {} " \
            "--source-key {} --recursive "
    if overwrite_if_exist:
        cmd += "--quiet "
    if verbose:
        cmd += "--verbose "
    cmd = cmd.format(storage_account.account_name, storage_account.container_name,
            azure_file_or_path, local_file_or_path, storage_account.account_key)
    print(cmd)
    os.system(cmd)


def transfer_files_between_azure_storages(source_file_or_path, dest_file_or_path,
        source_datastore_cfg, dest_datastore_cfg, overwrite_if_exist=True,
        verbose=False):
    source_storage_account = StorageAccount(source_datastore_cfg)
    dest_storage_account = StorageAccount(dest_datastore_cfg)
    cmd = "azcopy --source https://{}.blob.core.windows.net/{}/{} " \
            "--destination https://{}.blob.core.windows.net/{}/{} " \
            "--source-key {} --dest-key {} --recursive "
    if overwrite_if_exist:
        cmd += "--quiet "
    if verbose:
        cmd += "--verbose"
    cmd = cmd.format(source_storage_account.account_name,
            source_storage_account.container_name,
            source_file_or_path,
            dest_storage_account.account_name,
            dest_storage_account.container_name,
            dest_file_or_path,
            source_storage_account.account_key,
            dest_storage_account.account_key
    )
    print(cmd)
    os.system(cmd)

