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
sys.path.append(os.path.abspath('.'))
from utils.log_util import get_logger
import logging


logger = get_logger(name=__name__, log_file=Path(__file__).with_suffix('.log'), log_level=logging.INFO)


data_dir = Path('data')
data_file = data_dir / 'demo_train.h5ad'
ref_adata = sc.read(data_dir / 'demo_train.h5ad')
logger.info('ref_adata.var_names[:5] %s', ref_adata.var_names[:5])
logger.info('ref_adata.obs_names[:5] %s', ref_adata.obs_names[:5])
# view of the data
ref_adata = ref_adata[:, ref_adata.var_names]
logger.info('ref_adata %s', ref_adata)
logger.info('ref_adata.obs.Celltype.value_counts() %s', ref_adata.obs.Celltype.value_counts())

logger.info('*** show a short snippet of data')
a = ref_adata[['human1_lib1.final_cell_0001', 'human1_lib1.final_cell_0003'], ['COL1A1', 'COL1A2', 'PPY', 'CTRB1']]
logger.info(a)
dense_data = a.X.toarray().tolist()
logger.info(dense_data)

logger.info('*** query_adata')
query_adata = sc.read(data_dir / 'demo_test.h5ad')
query_adata = query_adata[:,ref_adata.var_names]
logger.info(query_adata)
logger.info(query_adata.obs.Celltype.value_counts())
logger.info(query_adata.var['Gene Symbol'])
