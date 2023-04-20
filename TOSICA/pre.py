import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import scanpy as sc
import anndata as ad
from .TOSICA_model import scTrans_model as create_model
from utils.log_util import logger
from torch.utils.data import DataLoader


#model_weight_path = "./weights20220429/model-5.pth"
#mask_path = os.getcwd()+'/mask.npy'

def todense(adata):
    import scipy
    if isinstance(adata.X, scipy.sparse.csr_matrix):
        return adata.X.todense()
    else:
        return adata.X

def get_weight(att_mat,pathway):
    att_mat = torch.stack(att_mat).squeeze(1)
    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    v = pd.DataFrame(v[0,1:].detach().numpy()).T
    #logger.info(v.size())
    v.columns = pathway
    return v

def prediect(adata, model_weight_path, project, mask_path, get_latent_output=False, 
             save_att='X_att', save_lantent='X_lat',
             n_step=10000, cutoff=0.1, n_unannotated=1, 
             batch_size=50, embed_dim=48, depth=2, num_heads=4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(device)
    num_genes = adata.shape[1]
    #mask_path = os.getcwd()+project+'/mask.npy'
    mask = np.load(mask_path)
    project_path = os.getcwd()+'/%s'%project
    # pathway column is '0' which is the default when save list as DataFrame.
    pathway = pd.read_csv(project_path+'/pathway.csv', index_col=0)
    dictionary = pd.read_table(project_path+'/label_dictionary.csv', sep=',',header=0,index_col=0)
    n_c = len(dictionary)
    label_name = dictionary.columns[0]
    dictionary.loc[(dictionary.shape[0])] = 'Unknown'
    dic = {}
    for i in range(len(dictionary)):
        dic[i] = dictionary[label_name][i]
    model = create_model(num_classes=n_c, num_genes=num_genes, mask=mask, has_logits=False,
                         depth=depth, num_heads=num_heads).to(device)
    # load model weights
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    parm={}
    for name,parameters in model.named_parameters():
        #logger.info(name,':',parameters.size())
        parm[name]=parameters.detach().cpu().numpy()
    gene2token = parm['feature_embed.fe.weight']
    # gene2token shape becomes (max_gs, embed_dim, num_genes)
    gene2token = gene2token.reshape((int(gene2token.shape[0]/embed_dim), embed_dim, num_genes))
    gene2token = abs(gene2token)
    gene2token = np.max(gene2token, axis=1)
    gene2token = pd.DataFrame(gene2token)
    gene2token.columns = adata.var_names
    gene2token.index = pathway['0']
    gene2token.to_csv(project_path+'/gene2token_weights.csv')
    latent = torch.empty([0, embed_dim]).cpu()
    att = torch.empty([0, len(pathway)]).cpu()
    predict_class = np.empty(shape=0)
    probabilities = np.empty(shape=0)
    latent = torch.squeeze(latent).cpu().numpy()
    # l_p shape: (0, embed_dim+2), att shape: (0, len(pathway)+2)
    l_p = np.c_[latent, predict_class, probabilities]
    att = np.c_[att, predict_class, probabilities]
    all_line = adata.shape[0]
    n_line = 0
    adata_list = []
    while (n_line) <= all_line:
        if (all_line-n_line)%batch_size != 1:
            start_i = n_line
            end_i = n_line + min(n_step,(all_line-n_line))
            sub_data = adata[start_i: end_i]
            expdata = pd.DataFrame(todense(sub_data), index=np.array(sub_data.obs_names).tolist(),
                                   columns=np.array(adata.var_names).tolist())
            logger.info(n_line)
            n_line = n_line+n_step
        else:
            expdata = pd.DataFrame(todense(adata[n_line:n_line+min(n_step,(all_line-n_line-2))]),index=np.array(adata[n_line:n_line+min(n_step,(all_line-n_line-2))].obs_names).tolist(), columns=np.array(adata.var_names).tolist())
            n_line = (all_line-n_line-2)
            logger.info(n_line)
        expdata = np.array(expdata)
        expdata = torch.from_numpy(expdata.astype(np.float32))
        data_loader = DataLoader(expdata, batch_size=batch_size, shuffle=False, pin_memory=True)
        with torch.no_grad():
            # predict class
            for step, data in enumerate(data_loader):
                #logger.info(step)
                exp = data
                lat, pre, weights = model(exp.to(device))
                pre = torch.squeeze(pre).cpu()
                pre = F.softmax(pre, 1)
                predict_class = np.empty(shape=0)
                probabilities = np.empty(shape=0)
                # TODO dqc ugly and bad duplicate logic
                for i in range(len(pre)):
                    if torch.max(pre, dim=1)[0][i] >= cutoff:
                        predict_class = np.r_[predict_class,torch.max(pre, dim=1)[1][i].numpy()]
                    else:
                        # n_c = len(dictionary), i.e. invalid label, because the max probability is less than cutoff. 
                        predict_class = np.r_[predict_class,n_c]
                    probabilities = np.r_[probabilities,torch.max(pre, dim=1)[0][i]]
                l_p = torch.squeeze(lat).cpu().numpy()
                att = torch.squeeze(weights).cpu().numpy()
                meta = np.c_[predict_class,probabilities]
                meta = pd.DataFrame(meta)
                meta.columns = ['Prediction','Probability']
                meta.index = meta.index.astype('str')
                if get_latent_output:
                    l_p = l_p.astype('float32')
                    new = sc.AnnData(l_p, obs=meta)
                else:
                    # there is n_unannotated and so minus it.
                    att = att[:,0:(len(pathway)-n_unannotated)]
                    att = att.astype('float32')
                    var_max_index = len(pathway)-1
                    varinfo = pd.DataFrame(pathway.iloc[:var_max_index, 0].values, 
                                           index=pathway.iloc[:var_max_index, 0],
                                           columns=['pathway_index'])
                    new = sc.AnnData(att, obs=meta, var = varinfo)
                adata_list.append(new)
    logger.info(all_line)
    new = ad.concat(adata_list)
    new.obs.index = adata.obs.index
    new.obs['Prediction'] = new.obs['Prediction'].map(dic)
    new.obs[adata.obs.columns] = adata.obs[adata.obs.columns].values
    return new
