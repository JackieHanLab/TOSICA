import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import scanpy as sc
from .TOSICA_model import scTrans_model as create_model



#model_weight_path = "./weights20220429/model-5.pth" 
mask_path = os.getcwd()+'/mask.npy'

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
    #print(v.size())
    v.columns = pathway
    return v

def prediect(adata,model_weight_path,mask_path = mask_path,laten=False,save_att = 'X_att', save_lantent = 'X_lat',n_step=10000,cutoff=0.1,n_unannotated = 1,batch_size = 50,embed_dim=48,depth=2,num_heads=4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    num_genes = adata.shape[1]
    mask = np.load(mask_path) 
    pathway = pd.read_csv(os.getcwd()+'/pathway.csv', index_col=0)
    dictionary = pd.read_table(os.getcwd()+'/label_dictionary.csv', sep=',',header=0,index_col=0)
    n_c = len(dictionary)
    model = create_model(num_classes=n_c, num_genes=num_genes,mask = mask, has_logits=False,depth=depth,num_heads=num_heads).to(device)
    # load model weights
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    parm={}
    for name,parameters in model.named_parameters():
        #print(name,':',parameters.size())
        parm[name]=parameters.detach().cpu().numpy()
    gene2token = parm['feature_embed.fe.weight']
    gene2token = gene2token.reshape((int(gene2token.shape[0]/embed_dim),embed_dim,adata.shape[1]))
    gene2token = abs(gene2token)
    gene2token = np.max(gene2token,axis=1)
    gene2token = pd.DataFrame(gene2token)
    gene2token.columns=adata.var_names
    gene2token.index = pathway['0']
    gene2token.to_csv('gene2token_weights.csv')
    latent = torch.empty([0,embed_dim]).cpu()
    att = torch.empty([0,(len(pathway))]).cpu()
    predict_class = np.empty(shape=0)
    pre_class = np.empty(shape=0)      
    latent = torch.squeeze(latent).cpu().numpy()
    l_p = np.c_[latent, predict_class,pre_class]
    att = np.c_[att, predict_class,pre_class]
    all_l_p = pd.DataFrame(l_p)
    all_att = pd.DataFrame(att)
    all_l_p.rename(columns={embed_dim:'Prediction',embed_dim+1:'Probability'},inplace = True)
    all_att.rename(columns={len(pathway):'Prediction',len(pathway)+1:'Probability'},inplace = True)
    all_line = adata.shape[0]
    #all_line = int(all_line.split()[0])
    n_line = 0
    while (n_line) <= all_line:
        if (all_line-n_line)%batch_size != 1:
            expdata = pd.DataFrame(todense(adata[n_line:n_line+min(n_step,(all_line-n_line))]),index=np.array(adata[n_line:n_line+min(n_step,(all_line-n_line))].obs_names).tolist(), columns=np.array(adata.var_names).tolist())
            print(n_line)
            n_line = n_line+n_step
        else:
            expdata = pd.DataFrame(todense(adata[n_line:n_line+min(n_step,(all_line-n_line-2))]),index=np.array(adata[n_line:n_line+min(n_step,(all_line-n_line-2))].obs_names).tolist(), columns=np.array(adata.var_names).tolist())
            n_line = (all_line-n_line-2)
            print(n_line)
        expdata = np.array(expdata)
        expdata = torch.from_numpy(expdata.astype(np.float32))
        data_loader = torch.utils.data.DataLoader(expdata,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 pin_memory=True)
        with torch.no_grad():
            # predict class
            for step, data in enumerate(data_loader):
                #print(step)
                exp = data
                lat, pre, weights = model(exp.to(device))
                #weights = get_weight(weights,pathway)
                #att = torch.cat((att,weights),0)
                #latent = torch.cat((latent,lat),0)
                pre = torch.squeeze(pre).cpu()
                pre = F.softmax(pre,1)
                predict_class = np.empty(shape=0)
                pre_class = np.empty(shape=0) 
                for i in range(len(pre)):
                    if torch.max(pre, dim=1)[0][i] >= cutoff: 
                        predict_class = np.r_[predict_class,torch.max(pre, dim=1)[1][i].numpy()]
                    else:
                        predict_class = np.r_[predict_class,n_c]
                    pre_class = np.r_[pre_class,torch.max(pre, dim=1)[0][i]]     
                latent = torch.squeeze(lat).cpu().numpy()
                weights = torch.squeeze(weights).cpu().numpy()
                l_p = np.c_[latent, predict_class,pre_class]
                att = np.c_[weights, predict_class,pre_class]
                l_p = pd.DataFrame(l_p)
                att = pd.DataFrame(att)
                l_p.rename(columns={embed_dim:'Prediction',embed_dim+1:'Probability'},inplace = True)
                att.rename(columns={(len(pathway)):'Prediction',(len(pathway)+1):'Probability'},inplace = True)
                all_att = pd.concat([all_att,att],ignore_index = True)
                all_l_p = pd.concat([all_l_p,l_p],ignore_index = True)
                #l_p.to_csv(dataset+'_pre_'+str(cutoff)+'.csv',mode='a', header=0,index=False) 
                #att.to_csv(dataset+'_att_'+str(cutoff)+'.csv',mode='a', header=0,index=False)
    print(all_line)
    dictionary = pd.read_table(os.getcwd()+'/label_dictionary.csv', sep=',',header=0,index_col=0) 
    label_name = dictionary.columns[0]
    dictionary.loc[(dictionary.shape[0])] = 'Unknown'
    dic = {}
    for i in range(len(dictionary)):
        dic[i] = dictionary[label_name][i]
    if laten:
        all_l_p['Prediction'] = all_l_p['Prediction'].map(dic)
        meta = pd.DataFrame(all_l_p[["Prediction",'Probability']])
        all_l_p = all_l_p.iloc[:,0:embed_dim]
        new = sc.AnnData(all_l_p, obs=meta)
        new.obs[adata.obs.columns] = adata.obs[adata.obs.columns].values
        return(new)
    else:
        all_att['Prediction'] = all_att['Prediction'].map(dic)
        meta = pd.DataFrame(all_att[["Prediction",'Probability']])
        meta.index = adata.obs.index
        all_att = all_att.iloc[:,0:len(pathway)]
        all_att.columns = pathway.iloc[:,0]
        varinfo = pd.DataFrame(all_att.columns.values,index=all_att.columns,columns=['pathway_index'])
        new = sc.AnnData(all_att, obs=meta, var = varinfo)
        new.obs[adata.obs.columns] = adata.obs[adata.obs.columns].values
        return(new)
