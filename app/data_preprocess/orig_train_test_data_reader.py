import re, random
from pathlib import Path
import json
import os, sys
import scanpy as sc
import logging
import numpy as np


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s %(lineno)d: %(message)s',
                    datefmt='%y-%m-%d %H:%M')
root_data_dir = Path('data')


def read_train_test_data(data_type, verbose=True):
    """
    ref_adata is train data, and query_adata is test data.
    Warning: the `var_names` (genes) of the `ref_adata` and `query_adata` must be consistent and in the same order.
    """
    data_dir = root_data_dir / data_type
    ref_adata = sc.read(data_dir / 'train.h5ad')
    ref_adata = ref_adata[:, ref_adata.var_names]

    query_adata = sc.read(data_dir / 'test.h5ad')

    assert np.all(ref_adata.var_names == query_adata.var_names)
    query_adata = query_adata[:, query_adata.var_names]

    if verbose:
        logger.info('ref_adata.var_names %s', ref_adata.var_names)
        logger.info('query_adata.var_names %s', query_adata.var_names)
        logger.info('ref_adata %s', ref_adata)
        logger.info('query_adata %s', query_adata)
    return ref_adata, query_adata


if __name__ == "__main__":
    read_train_test_data('hPancreas')