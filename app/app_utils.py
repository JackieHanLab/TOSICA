import re, random
from pathlib import Path
import json
import os, sys
from utils.file_util import FileUtil
from utils.log_util import logger


def read_train_config(data_type):
    """  """
    model_weight_paths = []
    logger.info('data_type %s', data_type)
    for file in Path('config').iterdir():
        if file.stem.startswith(data_type):
            configs = FileUtil.read_json(file)
            best_epoch = configs['best_epoch']
            model_dir_name = file.stem
            model_weight_path = f'model_files/{model_dir_name}/model-{best_epoch}.pth'
            model_weight_paths.append(model_weight_path)
    return model_weight_paths
