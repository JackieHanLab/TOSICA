import random
import numpy as np
from torch.utils.data import Dataset
import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import os
import json
import math
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import platform
from utils.log_util import logger
from .TOSICA_model import scTrans_model as create_model
from pathlib import Path
from sklearn.model_selection import train_test_split


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

def todense(adata):
    import scipy
    if isinstance(adata.X, scipy.sparse.csr_matrix):
        return adata.X.todense()
    else:
        return adata.X

class MyDataSet(Dataset):
    """
    Preproces input matrix and labels.

    """
    def __init__(self, exp, label):
        self.exp = exp
        self.label = label
        self.len = len(label)

    def __getitem__(self,index):
        return self.exp[index], self.label[index]

    def __len__(self):
        return self.len

def balance_populations(data):
    logger.info('%s', data.shape)
    data_label = data[:,-1]
    ct_names = np.unique(data_label)
    ct_counts = pd.value_counts(data_label)
    logger.info('ct_counts\n%s', ct_counts)
    logger.info('len(ct_counts) %s', len(ct_counts))
    logger.info('2000000/len(ct_counts) %s', 2000000/len(ct_counts))
    max_val = min(ct_counts.max(), np.int32(2000000/len(ct_counts)))
    logger.info('max_val %s', max_val)
    balanced_data = np.empty(shape=(1, data.shape[1]), dtype=np.float32)
    for ct in ct_names:
        tmp = data[data_label == ct]
        orig_num = len(tmp)
        if orig_num >= max_val:
            idx = np.random.choice(range(orig_num), max_val, replace=False)
        else:
            random_num = max_val - orig_num
            random_idx = np.random.choice(range(orig_num), random_num)
            basic_index = np.array(range(orig_num))
            idx = np.r_[random_idx, basic_index]
        tmp_X = tmp[idx]
        # the same as np.concatenate([balanced_data,tmp_X])
        balanced_data = np.r_[balanced_data, tmp_X]
    return np.delete(balanced_data, 0, axis=0)

def splitDataSet(adata, label_name='Celltype', data_seed=0, test_size=0.2):
    """
    Split data set into training set and test set.
    """
    label_encoder = LabelEncoder()
    el_data = pd.DataFrame(
        todense(adata), index=np.array(adata.obs_names).tolist(), columns=np.array(adata.var_names).tolist())
    logger.info('el_data.shape %s', el_data.shape)
    el_data[label_name] = adata.obs[label_name].astype('str')
    logger.info('el_data with label shape %s', el_data.shape)
    genes = el_data.columns.values[:-1]
    el_data = np.array(el_data)
    # el_data = np.delete(el_data,-1,axis=1)
    el_data[:, -1] = label_encoder.fit_transform(el_data[:, -1])
    inverse = label_encoder.inverse_transform(range(0, np.max(el_data[:, -1])+1))
    logger.info('label inverse %s', inverse)
    el_data = el_data.astype(np.float32)
    logger.info('orig el_data[:5]\n%s', el_data[:5])
    el_data = balance_populations(data = el_data)
    logger.info('After balance_populations el_data[:5]\n%s', el_data[:5])
    logger.info('After balance_populations, train+valid shape %s', el_data.shape)
    n_genes = len(el_data[1])-1
    train_dataset, valid_dataset = train_test_split(el_data, test_size=test_size, random_state=data_seed)
    exp_train = torch.from_numpy(train_dataset[:,:n_genes].astype(np.float32))
    label_train = torch.from_numpy(train_dataset[:,-1].astype(np.int64))
    exp_valid = torch.from_numpy(valid_dataset[:,:n_genes].astype(np.float32))
    label_valid = torch.from_numpy(valid_dataset[:,-1].astype(np.int64))
    logger.info('len(exp_train) %s', len(exp_train))
    logger.info('len(exp_valid) %s', len(exp_valid))
    logger.info('exp_train[:5]\n%s', exp_train[:5])
    logger.info('label_train[:10]\n%s', label_train[:10])
    return exp_train, label_train, exp_valid, label_valid, inverse, genes


def get_gmt(gmt):
    import pathlib
    root = pathlib.Path(__file__).parent
    gmt_files = {
        "human_gobp": [root / "resources/GO_bp.gmt"],
        "human_immune": [root / "resources/immune.gmt"],
        "human_reactome": [root / "resources/reactome.gmt"],
        "human_tf": [root / "resources/TF.gmt"],
        "mouse_gobp": [root / "resources/m_GO_bp.gmt"],
        "mouse_reactome": [root / "resources/m_reactome.gmt"],
        "mouse_tf": [root / "resources/m_TF.gmt"]
    }
    return gmt_files[gmt][0]

def read_gmt(fname, sep='\t', min_g=0, max_g=5000):
    """
    Read GMT file into dictionary of gene_module:genes.\n
    min_g and max_g are optional gene set size filters.

    Args:
        fname (str): Path to gmt file
        sep (str): Separator used to read gmt file.
        min_g (int): Minimum of gene members in gene module.
        max_g (int): Maximum of gene members in gene module.
    Returns:
        OrderedDict: Dictionary of gene_module:genes.
    """
    dict_pathway = OrderedDict()
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            val = line.split(sep)
            if min_g <= len(val[2:]) <= max_g:
                dict_pathway[val[0]] = val[2:]
    return dict_pathway

def create_pathway_mask(feature_list, dict_pathway, add_missing=1, fully_connected=True, to_tensor=False):
    """
    Creates a mask of shape [genes, pathways] where (i,j) = 1 if gene i is in pathway j, 0 else.

    Expects a list of genes and pathway dict.
    Note: dict_pathway should be an Ordered dict so that the ordering can be later interpreted.

    Args:
        feature_list (list): List of genes in single-cell dataset, e.g. 3000 gene names.
        dict_pathway (OrderedDict): Dictionary of gene_module: genes, e.g. 7000 pathways.
        add_missing (int): Number of additional, fully connected nodes.
        fully_connected (bool): Whether to fully connect additional nodes or not.
        to_tensor (False): Whether to convert mask to tensor or not.
    Returns:
        torch.tensor/np.array: Gene module mask.
    """
    assert type(dict_pathway) == OrderedDict
    gene_num = len(feature_list)
    p_mask = np.zeros((gene_num, len(dict_pathway)))
    pathways = list()
    for pathway_i, pathway in enumerate(dict_pathway.keys()):
        pathways.append(pathway)
        for i in range(gene_num):
            if feature_list[i] in dict_pathway[pathway]:
                p_mask[i, pathway_i] = 1.
    if add_missing:
        n = 1 if type(add_missing)==bool else add_missing
        # Get non connected genes
        if not fully_connected:
            idx_0 = np.where(np.sum(p_mask, axis=1)==0)
            vec = np.zeros((gene_num,n))
            vec[idx_0,:] = 1.
        else:
            vec = np.ones((gene_num, n))
        p_mask = np.hstack((p_mask, vec))
        for i in range(n):
            x = 'node %d' % i
            pathways.append(x)
    if to_tensor:
        p_mask = torch.Tensor(p_mask)
    return p_mask, np.array(pathways)

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Train the model and updata weights.
    """
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        exp, labels = data
        labels = labels.to(device)
        sample_num += exp.shape[0]
        _,pred,_ = model(exp.to(device))
        pred_classes = torch.argmax(pred, dim=1)
        accu_num += torch.eq(pred_classes, labels).sum()
        loss = loss_function(pred, labels)
        loss.backward()
        accu_loss += loss.detach()
        data_loader.desc = "[train epoch {}] loss: {:.4f}, acc: {:.4f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
        if not torch.isfinite(loss):
            logger.info('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0
    # data_loader = tqdm(data_loader)  # need tqdm for evaluation.
    # We just need a metric on the whole dataloader during evaluation.
    for step, data in enumerate(data_loader):
        exp, labels = data
        labels = labels.to(device)
        sample_num += exp.shape[0]
        _,pred,_ = model(exp.to(device))
        pred_classes = torch.argmax(pred, dim=1)
        accu_num += torch.eq(pred_classes, labels).sum()
        loss = loss_function(pred, labels)
        accu_loss += loss
        
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

def fit_model(
    adata, gmt_path, data_type, project_path:Path, 
    pre_weights='', label_name='Celltype', max_g=300, max_gs=300,
    mask_ratio=0.015,
    n_unannotated=1,
    batch_size=6,
    embed_dim=48,
    depth=2,
    num_heads=4,
    lr=0.001,
    epochs=10,
    seed=0,
    data_seed=0,
    lrf=0.01
):
    set_seed(seed)
    logger.info('seed %s', seed)
    # np.random is used to create mock data or split data
    np.random.seed(data_seed)
    device = 'cuda:0'
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(device)
    today = datetime.today().strftime('%y%m%d')
    #train_weights = os.getcwd()+"/weights%s"%today
    project_path = str(project_path)
    logger.info('project_path %s', project_path)
    tb_writer = SummaryWriter()
    exp_train, label_train, exp_valid, label_valid, inverse, genes = splitDataSet(adata, label_name, data_seed)
    if gmt_path is None:
        mask = np.random.binomial(1, mask_ratio, size=(len(genes), max_gs))
        pathways = list()
        for i in range(max_gs):
            x = 'node %d' % i
            pathways.append(x)
        logger.info('Full connection!')
    else:
        if '.gmt' in gmt_path:
            gmt_path = gmt_path
        else:
            gmt_path = get_gmt(gmt_path)
        reactome_dict = read_gmt(gmt_path, min_g=0, max_g=max_g)
        mask, pathways = create_pathway_mask(feature_list=genes,
                                          dict_pathway=reactome_dict,
                                          add_missing=n_unannotated,
                                          fully_connected=True)
        # Keeps pathways which have more than 4 genes
        pathways = pathways[np.sum(mask, axis=0)>4]
        # mask.shape e.g. (3000, 3479)
        mask = mask[:, np.sum(mask, axis=0)>4]
        # logger.info(mask.shape)
        selected_pathway_num = min(max_gs, mask.shape[1])
        # logger.info('selected_pathway_num %s', selected_pathway_num)
        sorted_pathway_index = np.argsort(np.sum(mask, axis=0))
        index_of_pathways_with_top_genes_num = sorted(sorted_pathway_index[-selected_pathway_num:])
        pathways = pathways[index_of_pathways_with_top_genes_num]
        mask = mask[:, index_of_pathways_with_top_genes_num]
        logger.info(mask.shape)
        logger.info('Mask loaded!')
    # mask is a (genes, mpathways) mapping matrix
    np.save(project_path+'/mask.npy', mask)
    pd.DataFrame(pathways).to_csv(project_path+'/pathway.csv')
    pd.DataFrame(inverse, columns=[label_name]).to_csv(project_path+'/label_dictionary.csv', quoting=None)
    num_classes = np.int64(torch.max(label_train)+1)
    #logger.info(num_classes)
    train_dataset = MyDataSet(exp_train, label_train)
    valid_dataset = MyDataSet(exp_valid, label_valid)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True, drop_last=False)
    model = create_model(
        num_classes=num_classes, num_genes=len(genes), mask=mask, embed_dim=embed_dim,
        depth=depth, num_heads=num_heads, has_logits=False).to(device)
    if pre_weights != "":
        assert os.path.exists(pre_weights), "pre_weights file: '{}' not exist.".format(pre_weights)
        preweights_dict = torch.load(pre_weights, map_location=device)
        logger.info(model.load_state_dict(preweights_dict, strict=False))
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        logger.info(name)
    logger.info('Model builded!')
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    early_stop_acc = 0.9999
    least_val_loss = float('inf')
    model_path = f'model_files/{data_type}-{today}-seed{seed}'
    Path(model_path).mkdir(exist_ok=1)
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        scheduler.step()
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=valid_loader,
                                     device=device,
                                     epoch=epoch)
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        logger.info('train_loss %s train_acc, %s val_loss %s val_acc, %s at epoch %s',
                    train_loss, train_acc, val_loss, val_acc, epoch)
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        if val_loss < least_val_loss:
            least_val_loss = val_loss
            best_epoch = epoch
            val_acc_at_least_val_loss = val_acc
            train_acc_at_least_val_loss = train_acc
            if platform.system().lower() == 'windows':
                torch.save(model.state_dict(), model_path+"/model-{}.pth".format(epoch))
            else:
                torch.save(model.state_dict(), model_path+"/model-{}.pth".format(epoch))
            logger.info('least_val_loss %s at best_epoch %s, val_acc_at_least_val_loss %s',
                        least_val_loss, best_epoch, val_acc_at_least_val_loss)
        if train_acc >= early_stop_acc:
            logger.info('Average train_acc %s >= early_stop_acc {} at epoch %d, so early stop',
                        train_acc, early_stop_acc, epoch)
            break
    
    train_config_file = f'config/{data_type}-{today}-seed{seed}.json'
    save_train_configs(
        train_config_file, project_path=project_path, pre_weights=pre_weights, label_name=label_name,
        max_g=max_g, max_gs=max_gs, mask_ratio=mask_ratio, n_unannotated=n_unannotated, batch_size=batch_size,
        embed_dim=embed_dim, depth=depth, num_heads=num_heads,lr=lr, epochs= epochs, seed=seed, lrf=lrf,
        data_seed=data_seed,
        least_val_loss=least_val_loss,
        best_epoch=best_epoch,
        val_acc_at_least_val_loss=val_acc_at_least_val_loss,
        train_acc_at_least_val_loss=train_acc_at_least_val_loss,
        )
    logger.info('Training finished!')


def save_train_configs(train_config_file, **kw):
    """  """
    with open(train_config_file, 'w', encoding='utf-8') as f:
        json.dump(kw, f, ensure_ascii=False, indent=4)
