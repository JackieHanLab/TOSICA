import json
import dataclasses
from decimal import Decimal
import dataclasses, json
from typing import Tuple, List
import numpy
import math
import hashlib
import socket
import os, sys
sys.path.append(os.path.abspath('.'))
from utils.log_util import logger
from functools import cache


class FileUtil(object):
    """
    文件工具类
    """
    @classmethod
    def read_raw_text(cls, file_path) ->List[str]:
        """
        读取原始文本数据，每行均为纯文本
        """
        all_raw_text_list = []
        with open(file_path, "r", encoding="utf-8") as raw_text_file:
            for item in raw_text_file:
                item = item.strip()
                all_raw_text_list.append(item)

        return all_raw_text_list

    @classmethod
    def write_raw_text(cls, texts, file_path):
        """
        写入文本数据，每行均为纯文本
        """
        with open(file_path, "w", encoding="utf-8") as f:
            for item in texts:
                f.write(f'{item}\n')

    @classmethod
    def read_json(cls, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    @classmethod
    def write_json(cls, data, file_path, ensure_ascii=False):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=ensure_ascii, indent=4, cls=JSONEncoder)


def dataclass_from_dict(klass, dikt):
    try:
        fieldtypes = klass.__annotations__
        return klass(**{f: dataclass_from_dict(fieldtypes[f], dikt[f]) for f in dikt})
    except AttributeError:
        # Must to support List[dataclass]
        if isinstance(dikt, (tuple, list)):
            return [dataclass_from_dict(klass.__args__[0], f) for f in dikt]
        return dikt


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, (Tuple, set)):
            return list(o)
        if isinstance(o, bytes):
            return o.decode()
        if isinstance(o, numpy.ndarray):
            return o.tolist()
        return super().default(o)


def get_partial_files(input_files, total_parts_num=-1, part_num=-1, start_index=-1) ->List:
    """ part_num starts from 1.
        If set start_index > 0, directly get partial input_files[start_index:]
    """
    if start_index > 0:
        partial_files = input_files[start_index:]
    elif part_num > 0 and total_parts_num > 0:
        input_files_num = len(input_files)
        num_per_part = math.ceil(input_files_num / total_parts_num)
        start_i = (part_num - 1) * num_per_part
        end_i = part_num * num_per_part
        partial_files = input_files[start_i: end_i]
    return partial_files


def calculate_file_md5(filename):
    """ For small file """
    with open(filename,"rb") as f:
        bytes = f.read()
        readable_hash = hashlib.md5(bytes).hexdigest()
        return readable_hash


def calculate_file_md5_large_file(filename):
    """ For large file to read by chunks in iteration. """
    md5_hash = hashlib.md5()
    with open(filename,"rb") as f:
        # Read and update hash in chunks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
        return md5_hash.hexdigest()

@cache
def get_local_ip(only_last_address=True) -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('192.255.255.255', 1))
        local_ip = s.getsockname()[0]
    except Exception as identifier:
        logger.info('cannot get ip with error %s\nSo the local ip is 127.0.0.1', identifier)
        local_ip = '127.0.0.1'
    finally:
        s.close()
    logger.info('full local_ip %s, only_last_address %s', local_ip, only_last_address)
    if only_last_address:
        local_ip = local_ip.split('.')[-1]
    return local_ip
