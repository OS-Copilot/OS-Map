import os
from typing import Dict, List, Set
from typing import Optional, Any, Union
from datetime import datetime
import requests
import pandas as pd
import hashlib
from pathlib import Path


def get_content_from_vm_file(env, config: Dict[str, Any]) -> Any:
    """
    Config:
        path (str): absolute path on the VM to fetch
    """

    path = config["path"]
    file_path = get_vm_file(env, {"path": path, "dest": os.path.basename(path)})
    file_type, file_content = config['file_type'], config['file_content']
    if file_type == 'xlsx':
        if file_content == 'last_row':
            df = pd.read_excel(file_path)
            last_row = df.iloc[-1]
            last_row_as_list = last_row.astype(str).tolist()
            return last_row_as_list
    else:
        raise NotImplementedError(f"File type {file_type} not supported")


def get_cloud_file(env, config: Dict[str, Any]) -> Union[str, List[str]]:
    """
    Config:
        path (str|List[str]): the url to download from
        dest (str|List[str])): file name of the downloaded file
        multi (bool) : optional. if path and dest are lists providing
          information of multiple files. defaults to False
        gives (List[int]): optional. defaults to [0]. which files are directly
          returned to the metric. if len==1, str is returned; else, list is
          returned.
    """

    if not config.get("multi", False):
        paths: List[str] = [config["path"]]
        dests: List[str] = [config["dest"]]
    else:
        paths: List[str] = config["path"]
        dests: List[str] = config["dest"]
    cache_paths: List[str] = []

    gives: Set[int] = set(config.get("gives", [0]))

    for i, (p, d) in enumerate(zip(paths, dests)):
        _path = os.path.join(env.cache_dir, d)
        if i in gives:
            cache_paths.append(_path)

        if os.path.exists(_path):
            #return _path
            continue

        url = p
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    return cache_paths[0] if len(cache_paths)==1 else cache_paths


def get_vm_file(env, config: Dict[str, Any]) -> Union[Optional[str], List[Optional[str]]]:
    """
    Config:
        path (str): absolute path on the VM to fetch
        dest (str): file name of the downloaded file
        multi (bool) : optional. if path and dest are lists providing
          information of multiple files. defaults to False
        gives (List[int]): optional. defaults to [0]. which files are directly
          returned to the metric. if len==1, str is returned; else, list is
          returned.
        only support for single file now:
        time_suffix(bool): optional. defaults to False. if True, append the current time in required format.
        time_format(str): optional. defaults to "%Y_%m_%d". format of the time suffix.
    """
    time_format = "%Y_%m_%d"
    if not config.get("multi", False):
        paths: List[str] = [config["path"]]
        dests: List[str] = [config["dest"]]
        if "time_suffix" in config.keys() and config["time_suffix"]:
            if "time_format" in config.keys():
                time_format = config["time_format"]
            # Insert time before . in file type suffix
            paths = [p.split(".")[0] + datetime.now().strftime(time_format) + "." + p.split(".")[1] if "." in p else p for p in paths]
            dests = [d.split(".")[0] + datetime.now().strftime(time_format) + "." + d.split(".")[1] if "." in d else d for d in dests]
    else:
        paths: List[str] = config["path"]
        dests: List[str] = config["dest"]


    cache_paths: List[str] = []

    gives: Set[int] = set(config.get("gives", [0]))

    for i, (p, d) in enumerate(zip(paths, dests)):
        _path = os.path.join(env.cache_dir, d)
        file = env.controller.get_file(p)
        if file is None:
            #return None
            # raise FileNotFoundError("File not found on VM: {:}".format(config["path"]))
            if i in gives:
                cache_paths.append(None)
            continue

        if i in gives:
            cache_paths.append(_path)
        with open(_path, "wb") as f:
            f.write(file)
    return cache_paths[0] if len(cache_paths)==1 else cache_paths


def get_vm_dir(env, config):
    source_dir = config["path"]
    local_dir_name = Path(source_dir).name
    local_dir = os.path.join(env.cache_dir, local_dir_name)
    os.makedirs(local_dir, exist_ok=True)
    for node in env.controller.get_vm_directory_tree(source_dir)["children"]:
        if node["type"] != "file":
            continue
        get_vm_file(env, {"path": os.path.join(source_dir, node["name"]), "dest": os.path.join(local_dir_name, node["name"])})
    return local_dir


def get_vm_dir_file_hashes(env, config):
    def calc_file_hash(file_path, hash_algorithm='sha256'):
        hash_func = hashlib.new(hash_algorithm)
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192): 
                hash_func.update(chunk)
        return hash_func.hexdigest()
    
    local_dir = get_vm_dir(env, config)
    file_hashes = sorted([calc_file_hash(os.path.join(local_dir, file_path)) for file_path in os.listdir(local_dir)])
    return file_hashes


def get_cache_file(env, config: Dict[str, str]) -> str:
    """
    Config:
        path (str): relative path in cache dir
    """

    _path = os.path.join(env.cache_dir, config["path"])
    assert os.path.exists(_path)
    return _path


def get_file_unzipped(env, config):
    if config["path"].startswith("http"):
        path = get_cloud_file(env, config)
    else:
        path = get_vm_file(env, config)
    file_name = os.path.basename(path)
    assert file_name.endswith(".zip"), "Not a zip file!"
    os.system(f"cd {env.cache_dir} && unzip {file_name}")
    return os.path.join(env.cache_dir, file_name[:-4])
