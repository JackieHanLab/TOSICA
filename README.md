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

## 数据阅读

执行 tutorial.ipynb  
### 输入

1. 测序结果数据
```py
assert np.all(ref_adata.var_names == query_adata.var_names)
```
> Please run the code to make sure they are the same. var_names即测序矩阵中的基因的名称顺序

2. pathway数据
TOSICA/resources
代码见，get_gmt() in TOSICA/train.py
来自这个网站，https://www.gsea-msigdb.org/gsea/downloads.jsp，可以直接下载，即基因富集分析的基础数据。
https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2023.1.Hs/msigdb_v2023.1.Hs.xml.zip
3. 重要：训练样本的样本分布不均衡，数量太少的样本，模型肯定学习和预测的效果不太好。

## 训练和预测

sh run.sh
run.sh的参数之一，enable_train=0，即不训练，而只是推理预测。enable_train=1，训练。
训练和测试文件，data/hPancreas/

不使用GPU而使用CPU的简易方式，run.sh里面export CUDA_VISIBLE_DEVICES=1， 这个值大于等于GPU数量即可。比如GPU数量是1，这个值>=1，实际上禁用GPU。
hPancreas数据集测试：GPU训练时间21分钟，CPU训练时间7小时5分钟。
预测时间，由于预测数据集只有4200，无论GPU和CPU都是一分钟。
这个模型是不大的模型结构，并且hPancreas训练数据很小，其实CPU虽然慢一点，也可接受。hPancreas是很小数据集，14.8K数据，如果大数据集 如mAltas, 356K, 训练用时大概7*25小时，恐怕时间太长了。

### 预测注意

1. 预测时候，如果未知类型，可能预测为unknown，也可能预测为训练集中的细胞类型。

## 建议步骤

1. 执行 tutorial.ipynb
2. 理解 训练和测试文件，data/hPancreas/的格式
3. sh run.sh
run.sh的参数之一，enable_train=0， 推理预测。
4. 用其他工具预测，对比效果
5. 把自己的人类胰腺数据，能否修改data/hPancreas/的格式，再次运行预测
用其他工具预测，对比效果
6. 使用作者公开的其他数据集，训练和预测，验证
sh run.sh
run.sh的参数之一，enable_train=1， 训练。
https://github.com/JackieHanLab/TOSICA/issues/9

### Orig Example Demo

[Guided Tutorial](tutorial.ipynb)

### Cite TOSICA

[Chen, J., Xu, H., Tao, W. et al. Transformer for one stop interpretable cell type annotation. Nat Commun 14, 223 (2023).](https://doi.org/10.1038/s41467-023-35923-4)
