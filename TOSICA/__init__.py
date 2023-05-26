r"""
TOSICA (Transformer for One-Stop Interpretable Cell-type Annotation)
"""
from datetime import datetime
import json
from .train import fit_model
from .pre import prediect
import os
from pathlib import Path

name = "TOSICA"
# __version__ = 1.0


def train(
    adata, gmt_path, data_type, project_path, pre_weights='', label_name='Celltype',
    max_g=300, max_gs=300, mask_ratio=0.015, n_unannotated=1,
    batch_size=8, embed_dim=48, depth=2, num_heads=4, lr=0.001, epochs=10, seed=3, data_seed=0, lrf=0.01,
    ignore_gpu=False,
    val_data_ratio=0.2,):
    r"""
    Fit the model with reference data
    Parameters
    ----------
    adatas
        Single-cell datasets
    gmt_path
        The name (human_gobp; human_immune; human_reactome; human_tf; mouse_gobp; mouse_reactome and mouse_tf) or path of mask to be used.
    project_path
        The name of saved project_path
    pre_weights
        The path to the pre-trained weights. If pre_weights = '', the model will be trained from scratch.
    label_name
        The column name of the label you want to prediect. Should in adata.obs.columns.
    max_g
        The max of gene number belong to one pathway.
    max_gs
        The max of pathway/token number.
    mask_ratio
        The ratio of the connection reserved when there is no available mask.
    n_unannotated
        The number of fully connected tokens to be added.
    batch_size
        The number of cells for training in one epoch.
    embed_dim
        The dimension of pathway/token embedding.
    depth
        The number of multi-head self-attention layer.
    num_heads
        The number of head in one self-attention layer.
    lr
        Learning rate.
    epochs
        The number of epoch will be trained.
    lrf
        The hyper-parameter of Cosine Annealing.
    Returns
    -------
    ./mask.npy
        Mask matrix
    ./pathway.csv
        Gene set list
    ./label_dictionary.csv
        Label list
    ./weights20220603/
        Weights
    """
    fit_model(adata, gmt_path, data_type, project_path=project_path, pre_weights=pre_weights, label_name=label_name,
              max_g=max_g, max_gs=max_gs, mask_ratio=mask_ratio, n_unannotated=n_unannotated, batch_size=batch_size,
              embed_dim=embed_dim, depth=depth, num_heads=num_heads, lr=lr, epochs= epochs, 
              seed=seed, data_seed=data_seed, lrf=lrf,
              ignore_gpu=ignore_gpu,
              val_data_ratio=val_data_ratio)


def pre(adata, model_weight_path, project_path:Path,
        get_latent_output=False, save_att = 'X_att', save_lantent = 'X_lat',
        n_step=10000, cutoff=0.1, n_unannotated = 1, batch_size=50,
        embed_dim=48, depth=2, num_heads=4):
    r"""
    Prediect query data with the model and pre-trained weights.
    Parameters
    ----------
    adatas
        Query single-cell datasets.
    model_weight_path
        The path to the pre-trained weights.
    mask_path
        The path to the mask matrix.
    project_path
        The name of saved project_path
    get_latent_output
        Get laten output.
    save_att
        The name of the attention matrix to be added in the adata.obsm.
    save_lantent
        The name of the laten matrix to be added in the adata.obsm.
    max_gs
        The max of pathway/token number.
    n_step
        The number of cells load into memory at the same time.
    cutoff
        Unknown cutoff.
    n_unannotated
        The number of fully connected tokens to be added. Should be the same as train.
    batch_size
        The number of cells for training in one epoch.
    embed_dim
        The dimension of pathway/token embedding. Should be the same as train.
    depth
        The number of multi-head self-attention layer. Should be the same as train.
    num_heads
        The number of head in one self-attention layer. Should be the same as train.
    Returns
    -------
    adata
        adata.X : Attention matrix
        adata.obs['Prediction'] : Predicted labels
        adata.obs['Probability'] : Probability of the prediction
        adata.var['pathway_index'] : Gene set of each colume
    """
    mask_path = project_path / 'mask.npy'
    adata = prediect(adata, model_weight_path, project_path=project_path,
                     mask_path=mask_path,
                     get_latent_output=get_latent_output, save_att=save_att, save_lantent = save_lantent,
                     n_step=n_step, cutoff=cutoff, n_unannotated = n_unannotated, batch_size=batch_size,
                     embed_dim=embed_dim, depth=depth, num_heads=num_heads)
    return adata
