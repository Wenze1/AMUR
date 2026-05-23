import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph


class GraphLearner(nn.Module):
    
    def __init__(self, config, network, device):
        super(GraphLearner, self).__init__()
        self.hidden = config['embedding_size']
        self.drop = config['drop']
        self.adj_matrix = network.coalesce()
        self.device = device
        self.edge_scores = self.adj_matrix.values().float().to(self.device).unsqueeze(-1)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * self.hidden + 1, self.hidden),
            nn.GELU(),
            nn.Dropout(self.drop),
            nn.Linear(self.hidden, 1)
        )        
    def sample_gumbel(self, shape):
        """ sample from Gumbel(0,1) """
        U = torch.rand(shape)
        return -torch.log(-torch.log(U+1e-8)+1e-8).to(self.device)
    
        
    def gumbel_softmax_sample(self, logits, temperature):
        values = (logits / temperature).sigmoid()
        values_hard = torch.where(values <= 0.5, 0, values)
        values = (values_hard - values).detach() + values
        y = torch.sparse.FloatTensor(self.adj_matrix.indices(), values, self.adj_matrix.shape).to(self.device)
        return y, values
        
    def forward(self, node_embs, temperature):
        f1 = node_embs[self.adj_matrix.indices()[0]]
        f2 = node_embs[self.adj_matrix.indices()[1]]
        edge_input = torch.cat([f1, f2, self.edge_scores], dim=-1)
        att_log = self.edge_mlp(edge_input).squeeze(-1)
        graph, att = self.gumbel_softmax_sample(att_log, temperature)
        return graph.coalesce(), att

class AMUR(GeneralRecommender):
    def __init__(self, config, dataset):
        super(AMUR, self).__init__(config, dataset)
        self.sparse = True
        self.cl_loss = config['cl_ip_loss']
        self.ip_loss = config['cl_ip_loss']
        self.kl_loss = config['kl_loss']
        self.kl_loss2 = config['kl_loss']
        self.shared_cl_loss = self._config_get(config, 'shared_cl_loss', self.cl_loss)
        self.shared_ip_loss = self._config_get(config, 'shared_ip_loss', self.ip_loss)
        self.eps = self._config_get(config, 'eps', 1e-8)
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.res_conn = config['res_conn']
        self.dataset = config['dataset']
        self.uicl_loss = config['uicl_loss']
        self.build_item_graph = True
        self.kl_func = nn.KLDivLoss(reduction="batchmean")
        self.softmax = nn.Softmax(dim=-1)
        
        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        self.item_text_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        self.item_image_embedding = nn.Embedding(self.n_items, self.embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_image_embedding.weight)
        nn.init.xavier_uniform_(self.item_text_embedding.weight)
        

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(dataset_path, 'image_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        text_adj_file = os.path.join(dataset_path, 'text_adj_{}_{}.pt'.format(self.knn_k, self.sparse))

        self.mse = nn.MSELoss()
        self.norm_adj = self.get_adj_mat()
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.item_graph_dict = np.load(os.path.join(dataset_path, config['item_graph_dict_file']),
                                       allow_pickle=True).item()
        
        __, self.session_adj = self.get_session_adj()


        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse,
                                                       norm_type='sym')
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda()

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda()

        self.softmax = nn.Softmax(dim=-1)
        
        self.image_graph = None
        self.text_graph = None
        
        self.graphlearner_image = GraphLearner(config, self.image_original_adj, device=self.device)
        self.graphlearner_text = GraphLearner(config, self.text_original_adj, device=self.device)
        self.image_behavior_ref = self.get_behavior_edge_targets(self.image_original_adj)
        self.text_behavior_ref = self.get_behavior_edge_targets(self.text_original_adj)
        self.image_edge_rows = self.image_original_adj.coalesce().indices()[0].to(self.device)
        self.text_edge_rows = self.text_original_adj.coalesce().indices()[0].to(self.device)
        self.shared_gate_image = nn.Sequential(
            nn.Linear(3 * self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.Dropout(config['drop']),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.shared_gate_text = nn.Sequential(
            nn.Linear(3 * self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.Dropout(config['drop']),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        

    def _config_get(self, config, key, default):
        try:
            return config[key]
        except (KeyError, TypeError):
            return default

    def pre_epoch_processing(self):
        self.build_item_graph = True
    
    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_behavior_edge_targets(self, candidate_adj):
        candidate_adj = candidate_adj.coalesce()
        candidate_indices = candidate_adj.indices().detach().cpu()
        session_adj = self.session_adj.coalesce().detach().cpu()
        session_indices = session_adj.indices()
        session_values = session_adj.values()
        behavior_map = {
            (int(session_indices[0, idx]), int(session_indices[1, idx])): float(session_values[idx])
            for idx in range(session_values.numel())
        }
        raw_targets = [
            behavior_map.get((int(candidate_indices[0, idx]), int(candidate_indices[1, idx])), 0.0)
            for idx in range(candidate_indices.size(1))
        ]
        targets = torch.tensor(raw_targets, dtype=torch.float32, device=self.device)
        rows = candidate_adj.indices()[0].to(self.device)
        row_sum = torch.zeros(self.n_items, device=self.device).scatter_add_(0, rows, targets)
        row_count = torch.zeros(self.n_items, device=self.device).scatter_add_(0, rows, torch.ones_like(targets))
        uniform_targets = 1.0 / row_count[rows].clamp_min(1.0)
        normalized_targets = targets / (row_sum[rows] + self.eps)
        targets = torch.where(row_sum[rows] > 0, normalized_targets, uniform_targets)
        return targets.clamp_min(self.eps).detach()

    def behavior_edge_kl(self, edge_probs, edge_targets, edge_rows):
        edge_probs = edge_probs.clamp_min(self.eps)
        row_sum = torch.zeros(self.n_items, device=self.device).scatter_add_(0, edge_rows, edge_probs)
        prob_dist = edge_probs / (row_sum[edge_rows] + self.eps)
        target_dist = edge_targets.clamp_min(self.eps)
        return (prob_dist * torch.log((prob_dist + self.eps) / target_dist)).mean()
    def get_r(self, decay_interval=10, decay_r=0.1, current_epoch=0, init_r=0.9, final_r=0.5): # 10 0.1 
        r = init_r - current_epoch // decay_interval * decay_r 
        if r < final_r:
            r = final_r
        return r
    
    def left_normalize(self, mx): # torch sp tensor-> dense->torch sp tensor 
        rowsum = torch.sparse.sum(mx, 1)
        r_inv = torch.pow(rowsum.values(), -1)
        r_inv = r_inv.masked_fill_(r_inv == float('inf'), 0).detach().cpu()
        offset = torch.tensor([0])
        r_mat_inv = torch.sparse.spdiags(r_inv, offset, mx.shape).to(self.device)
        # print(r_mat_inv.requires_grad)
        mx = torch.sparse.mm(r_mat_inv, mx)
        return mx
    
    def sym_normalize(self, mx): # torch sp tensor-> dense->torch sp tensor 
        rowsum = torch.sparse.sum(mx, 1)
        r_inv = torch.pow(rowsum.values(), -0.5)
        r_inv = r_inv.masked_fill_(r_inv == float('inf'), 0).detach().cpu()
        offset = torch.tensor([0])
        r_mat_inv = torch.sparse.spdiags(r_inv, offset, mx.shape).to(self.device)
        # print(r_mat_inv.requires_grad)
        mx = torch.sparse.mm(r_mat_inv, mx)
        mx = torch.sparse.mm(mx, r_mat_inv)
        return mx
    
    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def get_session_adj(self):
        index_x = []
        index_y = []
        values = []
        for i in range(self.n_items):
            index_x.append(i)
            index_y.append(i)
            values.append(1)
            if i in self.item_graph_dict.keys():
                item_graph_sample = self.item_graph_dict[i][0]
                item_graph_weight = self.item_graph_dict[i][1]

                for j in range(len(item_graph_sample)):
                    index_x.append(i)
                    index_y.append(item_graph_sample[j])
                    values.append(item_graph_weight[j])
        index_x = torch.tensor(index_x, dtype=torch.long)
        index_y = torch.tensor(index_y, dtype=torch.long)
        indices = torch.stack((index_x, index_y), 0).to(self.device)
        # norm
        return indices, self.compute_normalized_laplacian(indices, (self.n_items, self.n_items))
    
    def forward(self, adj, train=False, build_item_graph=False):
        image_item_embeds = self.item_id_embedding.weight
        # image_item_embeds = self.item_image_embedding.weight
        text_item_embeds = self.item_id_embedding.weight
        # text_item_embeds = self.item_text_embedding.weight

        # User-Item View
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings
        
        # graph structure learner
        if build_item_graph:
            updated_image_g, att_i = self.graphlearner_image(image_item_embeds, temperature=1) # torch sp tensor
            updated_text_g, att_t = self.graphlearner_text(text_item_embeds, temperature=1)
            image_graph = updated_image_g
            self.image_graph = (self.res_conn * self.image_original_adj + (1 - self.res_conn) * image_graph).coalesce()
            text_graph = updated_text_g
            self.text_graph = (self.res_conn * self.text_original_adj + (1 - self.res_conn) * text_graph).coalesce()
          
        else:
            self.image_graph = self.image_graph.detach()
            self.text_graph = self.text_graph.detach()
                
        if self.sparse:
            for i in range(self.n_layers):
                image_item_embeds = torch.sparse.mm(self.image_graph, image_item_embeds)
        else:
            for i in range(self.n_layers):
                image_item_embeds = torch.mm(self.image_graph, image_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)
        if self.sparse:
            for i in range(self.n_layers):
                text_item_embeds = torch.sparse.mm(self.text_graph, text_item_embeds)
        else:
            for i in range(self.n_layers):
                text_item_embeds = torch.mm(self.text_graph, text_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)
        
        side_embeds = (0.1*image_embeds + 0.9*text_embeds)/3
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        image_gate = self.shared_gate_image(torch.cat([image_item_embeds, text_item_embeds, content_embeds_items], dim=-1))
        text_gate = self.shared_gate_text(torch.cat([text_item_embeds, image_item_embeds, content_embeds_items], dim=-1))
        shared_image_item_embeds = image_gate * image_item_embeds
        shared_text_item_embeds = text_gate * text_item_embeds

        all_embeds = content_embeds + side_embeds # content->id side->MM

        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        if train:
            return (all_embeddings_users, all_embeddings_items, side_embeds, content_embeds,
                    image_item_embeds, text_item_embeds, shared_image_item_embeds,
                    shared_text_item_embeds, att_i, att_t)

        return all_embeddings_users, all_embeddings_items

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def KL_loss(self, att, r):
        kl_loss = (1e-8 + att * torch.log(att/(1e-8+r)) + (1-att) * torch.log(1e-8+ (1-att)/(-r+1+1e-8))).mean()
        return kl_loss 

    def calculate_loss(self, interaction, epoch):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        (ua_embeddings, ia_embeddings, side_embeds, content_embeds, image_item_embeds,
         text_item_embeds, shared_image_item_embeds, shared_text_item_embeds,
         att_i, att_t) = self.forward(
            self.norm_adj, train=True, build_item_graph=self.build_item_graph)
        
        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        kl_loss = self.behavior_edge_kl(att_i, self.image_behavior_ref, self.image_edge_rows)
        kl_loss2 = self.behavior_edge_kl(att_t, self.text_behavior_ref, self.text_edge_rows)

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)
        
        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        cl_loss = 0.5 * (
            self.InfoNCE(shared_image_item_embeds[pos_items], shared_text_item_embeds[pos_items], 0.2) +
            self.InfoNCE(shared_text_item_embeds[pos_items], shared_image_item_embeds[pos_items], 0.2)
        )
        ip_loss = self.mse(shared_image_item_embeds[pos_items], shared_text_item_embeds[pos_items])
            
        uicl_loss = self.InfoNCE(side_embeds_items[pos_items], u_g_embeddings, 0.2)
        aux_loss = self.shared_cl_loss * cl_loss + self.shared_ip_loss * ip_loss + self.uicl_loss * uicl_loss + self.kl_loss * kl_loss + self.kl_loss2 * kl_loss2

        return batch_mf_loss + batch_emb_loss + batch_reg_loss + aux_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj, train=False, build_item_graph=True)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
