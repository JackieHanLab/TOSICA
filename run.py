import scanpy as sc
import numpy as np
import warnings 
warnings.filterwarnings ("ignore")
import re, random
from pathlib import Path
import json
import pandas as pd
from pandas import DataFrame
import os, sys
import torch
import TOSICA 
from utils.log_util import logger
from utils.arg_util import ArgparseUtil
from utils.file_util import FileUtil


root_data_dir = Path('data')
data_type = 'hPancreas'
data_dir = root_data_dir / data_type
data_file = data_dir / 'demo_train.h5ad'
ref_adata = sc.read(data_dir / 'demo_train.h5ad')
ref_adata = ref_adata[:, ref_adata.var_names]
query_adata = sc.read(data_dir / 'demo_test.h5ad')
query_adata = query_adata[:,ref_adata.var_names]


def read_train_config():
    """  """
    for file in Path('config').iterdir():
        if file.stem.startswith(data_type):
            configs = FileUtil.read_json(file)
            best_epoch = configs['best_epoch']
            model_dir_name = '-'.join(file.stem.split('-')[:2])
            model_weight_path = f'model_files/{model_dir_name}/model-{best_epoch}.pth'
            return model_weight_path


train = 1
if train:
    TOSICA.train(
        ref_adata, gmt_path='human_gobp', data_type=data_type, label_name='Celltype', epochs=30, project='hGOBP_demo')
else:
    model_weight_path = read_train_config()
    new_adata = TOSICA.pre(query_adata, model_weight_path=model_weight_path, project='hGOBP_demo')