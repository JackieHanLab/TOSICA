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
from icecream import ic
import TOSICA 


root_data_dir = Path('data')
data_type = 'hPancreas'
data_dir = root_data_dir / data_type
data_file = data_dir / 'demo_train.h5ad'
ref_adata = sc.read(data_dir / 'demo_train.h5ad')
ref_adata = ref_adata[:, ref_adata.var_names]
query_adata = sc.read(data_dir / 'demo_test.h5ad')
query_adata = query_adata[:,ref_adata.var_names]

TOSICA.train(ref_adata, gmt_path='human_gobp', label_name='Celltype', epochs=3, project='hGOBP_demo')