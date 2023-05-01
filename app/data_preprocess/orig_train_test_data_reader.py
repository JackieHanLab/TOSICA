import re, random
from pathlib import Path
import json
import os, sys
import scanpy as sc
from utils.log_util import logger


root_data_dir = Path('data')


def read_train_test_data(data_type):
    """ ref_data is train data """
    data_dir = root_data_dir / data_type
    ref_data = sc.read(data_dir / 'train.h5ad')
    ref_data = ref_data[:, ref_data.var_names]

    query_adata = sc.read(data_dir / 'test.h5ad')
    query_adata = query_adata[:, query_adata.var_names]
    
    logger.info('ref_data %s', ref_data)
    logger.info('query_adata %s', query_adata)
    return ref_data, query_adata
