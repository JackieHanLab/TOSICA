# TOSICA: Transformer for One-Stop Interpretable Cell-type Annotation

![Workflow](./figure.png)

## Package: `TOSICA`

We created the python package called `TOSICA` that uses `scanpy` ans `torch` to explainablely annotate cell type on single-cell RNA-seq data.

### Requirements

+ Linux system
+ Python 3.9
+ torch 1.12.1

### Create environment

```bash
conda create -n tosica python=3.9 scanpy
conda activate tosica
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

### Environment has been tested

`TOSICA.yaml`

## Usage

### Step 1: Training the model

```bash
enable_train=1
run.sh
```

#### Input

+ `data_type`: each data_type links to relative ref_adata and query_adata(that's train/test data), used for train/test.
+ `gmt_path` : default pre-prepared mask or path to .gmt files.
+ `<my_project>`: project is different with data_type, mostly used for reference.
+ `<label_key>`: the name of the label column in `ref_adata.obs`.

#### Pre-prepared mask

+ `human_gobp` : GO_bp.gmt
+ `human_immune` : immune.gmt
+ `human_reactome` : reactome.gmt
+ `human_tf` : TF.gmt
+ `mouse_gobp` : m_GO_bp.gmt
+ `mouse_reactome` : m_reactome.gmt
+ `mouse_tf` : m_TF.gmt

#### Output

config files

+ `./projects/my_project/mask.npy` : Mask matrix
+ `./projects/my_project/pathway.csv` : Gene set list
+ `./projects/my_project/label_dictionary.csv` : Label list

saved model files

+ `./model_files/data_type/model-n.pth` : Weights
+ `./config/datatype-date-seed.json`: the most important model configs, including the best trained model epoch

### Step 2: Prediect by the model

```bash
enable_train=0
run.sh
```

#### Input

+ `data_type`: each data_type links to relative ref_adata and query_adata(that's train/test data), used for train/test.
+ `model_weight_path`: auto read from the json file in the "config" dir.
+ `project`: name of the folder build in training step, like: `my_project` or `<gmt_path>_20xxxxxx`.

#### Output

+ `./projects/my_project/predicted_result.h5ad`, this is an annData file, that's new_adata, `new_adata.X` : Attention matrix
+ `new_adata.obs['Prediction']` : Predicted labels
+ `new_adata.obs['Probability']` : Probability of the prediction
+ `new_adata.var['pathway_index']` : Gene set of each colume
+ `./projects/my_project/gene2token_weights.csv` : The weights matrix of genes to tokens

> **Warning:** the `var_names` (genes) of the `ref_adata` and `query_adata` must be consistent and in the same order.

```py
assert np.all(ref_adata.var_names == query_adata.var_names)
```

> Please run the code to make sure they are the same.  

### Example Demo

[Guided Tutorial](tutorial.ipynb)

### Cite TOSICA

[Chen, J., Xu, H., Tao, W. et al. Transformer for one stop interpretable cell type annotation. Nat Commun 14, 223 (2023).](https://doi.org/10.1038/s41467-023-35923-4)
