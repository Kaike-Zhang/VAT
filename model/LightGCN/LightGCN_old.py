import numpy as np
import scipy.sparse as sp
import dgl
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, normal_


class LightGCN(nn.Module):
    def __init__(self, config, adj_mat):
        super(LightGCN, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(self.config["n_users"] + self.config["n_items"] + 1, self.config["dim"], padding_idx=self.config["n_users"])
        self.norm_adj = self._generate_graph(adj_mat)
        self.n_layers = self.config["n_layers"] 
        self.device = self.config["device"]
        normal_(self.embedding.weight.data)

    def _get_rep(self):
        representations = self.embedding.weight
        all_layer_rep = [representations]
        row, column = self.norm_adj.indices()
        g = dgl.graph((column, row), num_nodes=self.norm_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=self.norm_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def forward(self, user_list, pos_items, neg_items):
        all_representation = self._get_rep()
        user_emb = self.embedding(torch.LongTensor(user_list).to(self.config["device"]))
        posI_emb = self.embedding(torch.LongTensor(pos_items).to(self.config["device"]) + self.config["n_users"])
        negI_emb = self.embedding(torch.LongTensor(neg_items).to(self.config["device"]) + self.config["n_users"])
        reg = (user_emb.norm(dim=1).pow(2).mean() + posI_emb.norm(dim=1).pow(2).mean() + negI_emb.norm(dim=1).pow(2).mean())

        return all_representation[torch.LongTensor(user_list).to(self.config["device"]), :], \
               all_representation[torch.LongTensor(pos_items).to(self.config["device"]) + self.config["n_users"], :], \
               all_representation[torch.LongTensor(neg_items).to(self.config["device"]) + self.config["n_users"], :], reg
    
    def predict(self, user_list):
        all_representation = self._get_rep()
        user_emb = all_representation[torch.LongTensor(user_list).to(self.config["device"]), :]
        item_emb = all_representation[self.config["n_users"]:, :]
        return torch.mm(user_emb, item_emb.t())

    def _generate_graph(self, adj_mat):
        degree = np.array(np.sum(adj_mat, axis=1)).squeeze()
        degree = np.maximum(1., degree)
        d_inv = np.power(degree, -0.5)
        d_mat = sp.diags(d_inv, format='csr', dtype=np.float32)

        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        norm_adj = self._get_sparse_tensor(norm_adj)
        return norm_adj
    
    def _get_sparse_tensor(self, mat):
        coo = mat.tocoo()
        indexes = np.stack([coo.row, coo.col], axis=0)
        indexes = torch.tensor(indexes, dtype=torch.int64, device=self.config["device"])
        data = torch.tensor(coo.data, dtype=torch.float32, device=self.config["device"])
        sp_tensor = torch.sparse.FloatTensor(indexes, data, torch.Size(coo.shape)).coalesce()
        return sp_tensor
