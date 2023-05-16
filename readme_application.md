# simple application this approach

## 数据阅读和初步体验
执行 tutorial.ipynb
目前，只执行到如下模板，后续画图，没有执行成功，不知道生信能否帮忙执行后续步骤？
new_adata.raw = new_adata
sc.pp.normalize_total(new_adata, target_sum=1e4)
sc.pp.log1p(new_adata)
sc.pp.scale(new_adata, max_value=10)
sc.tl.pca(new_adata, svd_solver='arpack')
sc.pp.neighbors(new_adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(new_adata)

### 训练和预测
sh run.sh
run.sh的参数之一，enable_train=0，即不训练，而只是推理预测。enable_train=1，训练。
训练和测试文件，data/hPancreas/

## 建议步骤：
1. 执行 tutorial.ipynb
2. 理解 训练和测试文件，data/hPancreas/的格式
3. sh run.sh
run.sh的参数之一，enable_train=0， 推理预测。
4. 用其他工具预测，对比效果
5. 把自己的人类胰腺数据，能否修改data/hPancreas/的格式，再次运行预测
用其他工具预测，对比效果
6. 使用作者公开的其他数据集，训练和预测，验证
https://github.com/JackieHanLab/TOSICA/issues/9