import re, random
from pathlib import Path
import json
import pandas as pd
import numpy as np
from pandas import DataFrame
import os, sys
from utils.log_util import logger


UNKNOWN = 'Unknown'
# label_name = 'Celltype'


def calc_accuracy(project_path, query_adata, new_adata):
    """  """
    dictionary = pd.read_table(project_path / 'label_dictionary.csv', sep=',', header=0, index_col=0)
    logger.info('%s', dictionary.shape)
    train_types = dictionary.iloc[:, 0].values.tolist()
    train_types.append(UNKNOWN)
    logger.info('train_types %s', train_types)

    query_types = query_adata.obs.Celltype.unique().tolist()
    logger.info('query_types %s', query_types)
    query_new_types = {}
    for query_type in query_types:
        if query_type not in train_types:
            query_new_types[query_type] = UNKNOWN
    logger.info('query_new_types %s', query_new_types)

    orig_true_labels = query_adata.obs.Celltype.to_numpy()
    for i, true_label in enumerate(orig_true_labels):
        if true_label not in train_types:
            orig_true_labels[i] = UNKNOWN
    logger.info('orig_true_labels[:5] %s', orig_true_labels[:5])

    logger.info('new_adata.obs.columns() %s', new_adata.obs.columns.tolist())
    pred_labels = new_adata.obs['Prediction'].to_numpy()
    logger.info('pred_labels[:5] %s', pred_labels[:5])

    correct_count = sum(orig_true_labels == pred_labels)
    accuracy = correct_count / len(pred_labels)
    logger.info('correct_count %s accuracy %s', correct_count, accuracy)
    