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
from app.projects.postprocess.calc_performance_on_query import calc_accuracy


def read_train_config():
    """  """
    for file in Path('config').iterdir():
        if file.stem.startswith(data_type):
            configs = FileUtil.read_json(file)
            best_epoch = configs['best_epoch']
            model_dir_name = file.stem
            model_weight_path = f'model_files/{model_dir_name}/model-{best_epoch}.pth'
            logger.info('trained model_weight_path %s', model_weight_path)
            return model_weight_path


args = ArgparseUtil().classifier()
data_type = args.data_type
root_data_dir = Path('data')
data_dir = root_data_dir / data_type
project = 'hGOBP_demo'
label_name = 'Celltype'
ref_adata = sc.read(data_dir / 'demo_train.h5ad')
ref_adata = ref_adata[:, ref_adata.var_names]
query_adata = sc.read(data_dir / 'demo_test.h5ad')
query_adata = query_adata[:, query_adata.var_names]

if args.enable_train:
    TOSICA.train(
        ref_adata, gmt_path='human_gobp', data_type=data_type, label_name=label_name,
        epochs=args.n_epoch, project=project,
        data_seed=args.data_seed,
        seed=args.seed)
else:
    read_cached_prediction = 1
    Path(f'cache/{project}').mkdir(exist_ok=1, parents=1)
    cached_prediction_file = Path(f'cache/{project}/prediction_file.h5ad')
    if read_cached_prediction and cached_prediction_file.is_file():
        new_adata = sc.read(cached_prediction_file)
        logger.info('Loads cached_prediction_file')
    else:
        model_weight_path = read_train_config()
        new_adata = TOSICA.pre(query_adata, model_weight_path=model_weight_path, project=project)
        new_adata.write(cached_prediction_file)
    calc_accuracy(project, query_adata, new_adata)