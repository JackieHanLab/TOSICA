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
from app.data_preprocess.orig_train_test_data_reader import read_train_test_data
from utils.file_util import FileUtil


logger = get_logger(name=__name__, log_file=Path(__file__).with_suffix('.log'), log_level=logging.INFO)
data_type = 'hPancreas'
ref_adata, query_adata = read_train_test_data(data_type, verbose=1)
special_genes = ['AP002495.1']


def basic_check():
    """  """
    logger.info('ref_data.var_names[:5] %s', ref_adata.var_names[:5])
    logger.info('ref_data.obs_names[:5] %s', ref_adata.obs_names[:5])
    # view of the data
    logger.info('ref_data.obs.Celltype.value_counts() %s', ref_adata.obs.Celltype.value_counts())

    logger.info('*** show a short snippet of data')
    a = ref_adata[['human1_lib1.final_cell_0001', 'human1_lib1.final_cell_0003'], ['COL1A1', 'COL1A2', 'PPY', 'CTRB1']]
    logger.info(a)
    dense_data = a.X.toarray().tolist()
    logger.info(dense_data)

    gene_names_file = Path(__file__).parent / f'{data_type}_genes_names.txt'
    FileUtil.write_raw_text(ref_adata.var_names, gene_names_file)
    for special_gene in special_genes:
        if special_gene in ref_adata.var_names.tolist():
            logger.info('special_gene %s in ref_adata.var_names', special_gene)

    logger.info('*** query_adata')
    logger.info(query_adata.obs.Celltype.value_counts())
    logger.info(query_adata.var['Gene Symbol'])


def check_query_data_in_ref_data():
    for q_i, query_row in enumerate(query_adata.X.toarray()):
        for r_i, ref_row in ref_adata.X.toarray():
            if np.allclose(query_row, ref_row):
                logger.info('q_i %s query_row %s exists in ref', q_i, query_row)


basic_check()
# check_query_data_in_ref_data()