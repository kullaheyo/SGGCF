import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import random

class Info(object):
    '''
    [FOR `utils.logger`]

    the base class that packing all hyperparameters and infos used in the related model
    '''

    def __init__(self, embedding_size, embed_L2_norm):
        assert isinstance(embedding_size, int) and embedding_size > 0
        self.embedding_size = embedding_size
        assert embed_L2_norm >= 0
        self.embed_L2_norm = embed_L2_norm

    def get_title(self):
        dct = self.__dict__
        if '_info' in dct:
            dct.pop('_info')
        return '\t'.join(map(lambda x: dct[x].get_title() if isinstance(dct[x], Info) else x, dct.keys()))

    def get_csv_title(self):
        return self.get_title().replace('\t', ', ')

    def __getitem__(self, key):
        if hasattr(self, '_info'):
            return self._info[key]
        else:
            return self.__getattribute__(key)

    def __str__(self):
        dct = self.__dict__
        if '_info' in dct:
            dct.pop('_info')
        return '\t'.join(map(str, dct.values()))

    def get_line(self):
        return self.__str__()

    def get_csv_line(self):
        return self.get_line().replace('\t', ', ')

def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt
    return graph

def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), 
                                          torch.Size(graph.shape))
    return graph


class SGGCF_Info(Info):
    def __init__(self, embedding_size, embed_L2_norm, mess_dropout, node_dropout, alpha, cl_reg, cl_2_reg, cl_temp, cl_2_temp, drop_rate, aug_type, num_layers, act=nn.LeakyReLU()):
        super().__init__(embedding_size, embed_L2_norm)
        self.act = act
        assert 1 > mess_dropout >= 0
        self.mess_dropout = mess_dropout
        assert 1 > node_dropout >= 0
        self.node_dropout = node_dropout
        assert isinstance(num_layers, int) and num_layers > 0
        self.num_layers = num_layers
        self.alpha = alpha
        self.cl_reg = cl_reg
        self.cl_2_reg = cl_2_reg
        self.cl_temp = cl_temp
        self.cl_2_temp = cl_2_temp
        self.drop_rate = drop_rate
        self.aug_type = aug_type

class SGGCF(nn.Module):
    def get_infotype(self):
        return SGGCF_Info

    def __init__(self, info, dataset, raw_graph, device):
        super().__init__()
        self.num_users = dataset.num_users
        self.num_groups = dataset.num_groups
        self.num_items = dataset.num_items
        self.device = device
        self.info = info
        self.embedding_size = info['embedding_size']

        self.items_feature = nn.Parameter(
            torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)
        self.users_feature = nn.Parameter(
            torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.groups_feature = nn.Parameter(
            torch.FloatTensor(self.num_groups, self.embedding_size))
        nn.init.xavier_normal_(self.groups_feature)
        self.epison = 1e-8

        # copy from info
        self.embed_L2_norm = info['embed_L2_norm']
        self.act = self.info.act
        self.num_layers = self.info.num_layers
        self.cl_temp = self.info.cl_temp
        self.cl_2_temp = self.info.cl_2_temp
        self.cl_reg = self.info.cl_reg
        self.cl_2_reg = self.info.cl_2_reg
        self.alpha = self.info.alpha
        self.drop_rate = self.info.drop_rate
        self.aug_type = self.info.aug_type

        # Dropouts
        self.mess_dropout = nn.Dropout(self.info.mess_dropout, True)
        self.node_dropout = nn.Dropout(self.info.node_dropout, True)
        
        
        
        gu_graph, ui_graph, gi_graph = raw_graph
        gu_graph_prime, ui_graph_prime, gi_graph_prime = self.perturb_adj(raw_graph, self.drop_rate, self.aug_type)
        
        
        
        GI_graph = sp.bmat([[sp.identity(gi_graph.shape[0])-sp.identity(gi_graph.shape[0]), gi_graph],
                                 [gi_graph.T, sp.identity(gi_graph.shape[1])-sp.identity(gi_graph.shape[1])]])
        self.gi_graph_tensor = to_tensor(laplace_transform(GI_graph)).to(device)
        
        
        UI_graph = sp.bmat([[sp.identity(ui_graph.shape[0])-sp.identity(ui_graph.shape[0]), ui_graph],
                                 [ui_graph.T, sp.identity(ui_graph.shape[1])-sp.identity(ui_graph.shape[1])]])
        self.ui_graph_tensor = to_tensor(laplace_transform(UI_graph)).to(device)
        
        
        GU_graph = sp.bmat([[sp.identity(gu_graph.shape[0])-sp.identity(gu_graph.shape[0]), gu_graph],
                                 [gu_graph.T, sp.identity(gu_graph.shape[1])-sp.identity(gu_graph.shape[1])]])
        self.gu_graph_tensor = to_tensor(laplace_transform(GU_graph)).to(device)
        
        
        whole_graph = sp.bmat([[sp.identity(gu_graph.shape[1])-sp.identity(gu_graph.shape[1]), gu_graph.T, ui_graph],
                                 [gu_graph, sp.identity(gi_graph.shape[0])-sp.identity(gi_graph.shape[0]),gi_graph-gi_graph],
                                 [ui_graph.T, gi_graph.T-gi_graph.T, sp.identity(gi_graph.shape[1])-sp.identity(gi_graph.shape[1])]])
        self.whole_graph_tensor = to_tensor(laplace_transform(whole_graph)).to(device)
        
        whole_graph_prime = sp.bmat([[sp.identity(gu_graph.shape[1])-sp.identity(gu_graph.shape[1]), gu_graph_prime.T, ui_graph_prime],
                                 [gu_graph_prime, sp.identity(gi_graph.shape[0])-sp.identity(gi_graph.shape[0]),gi_graph-gi_graph],
                                 [ui_graph_prime.T, gi_graph.T-gi_graph.T, sp.identity(gi_graph.shape[1])-sp.identity(gi_graph.shape[1])]])
        self.whole_graph_tensor_prime = to_tensor(laplace_transform(whole_graph_prime)).to(device)
        
        
    def perturb_adj(self, raw_graph, drop_rate, aug_type):


        gu_graph = raw_graph[0]
        ui_graph = raw_graph[1]
        gi_graph = raw_graph[2]


        if aug_type == 0:#node_dropout all 
            drop_user_idx = random.sample(list(range(self.num_users)), int(self.num_users * drop_rate))
            indicator_user = np.ones(self.num_users, dtype=np.float32)
            indicator_user[drop_user_idx] = 0.
            diag_indicator_user = sp.diags(indicator_user)
            gu_graph_new = gu_graph.dot(diag_indicator_user)
            ui_graph_new = diag_indicator_user.dot(ui_graph)

            return gu_graph_new, ui_graph_new, gi_graph

        if aug_type == 1:#node_dropout gu
            drop_user_idx = random.sample(list(range(self.num_users)), int(self.num_users * drop_rate))
            indicator_user = np.ones(self.num_users, dtype=np.float32)
            indicator_user[drop_user_idx] = 0.
            diag_indicator_user = sp.diags(indicator_user)
            gu_graph_new = gu_graph.dot(diag_indicator_user)

            return gu_graph_new, ui_graph, gi_graph

        if aug_type == 2:#node_dropout ui
            drop_user_idx = random.sample(list(range(self.num_users)), int(self.num_users * drop_rate))
            indicator_user = np.ones(self.num_users, dtype=np.float32)
            indicator_user[drop_user_idx] = 0.
            diag_indicator_user = sp.diags(indicator_user)
            ui_graph_new = diag_indicator_user.dot(ui_graph)

            return gu_graph, ui_graph_new, gi_graph
        
        if aug_type == 3: #edge_drop all
            row_idx_gu = gu_graph.nonzero()[0]
            col_idx_gu = gu_graph.nonzero()[1]
            keep_edge_idx_gu = random.sample(list(range(len(gu_graph.data))), int(len(gu_graph.data) * (1 - self.drop_rate)))
            group_np_gu = np.array(row_idx_gu)[keep_edge_idx_gu]
            user_np_gu = np.array(col_idx_gu)[keep_edge_idx_gu]
            ratings_gu = np.ones_like(user_np_gu, dtype=np.float32)
            gu_graph_new = sp.csr_matrix((ratings_gu, (group_np_gu, user_np_gu)), shape=(self.num_groups, self.num_users))
            row_idx = ui_graph.nonzero()[0]
            col_idx = ui_graph.nonzero()[1]
            keep_edge_idx = random.sample(list(range(len(ui_graph.data))), int(len(ui_graph.data) * (1 - self.drop_rate)))
            user_np = np.array(row_idx)[keep_edge_idx]
            item_np = np.array(col_idx)[keep_edge_idx]
            ratings = np.ones_like(user_np, dtype=np.float32)
            ui_graph_new = sp.csr_matrix((ratings, (user_np, item_np)), shape=(self.num_users, self.num_items))
            return gu_graph_new, ui_graph_new, gi_graph
        
        if aug_type == 4: #edge_drop ui
            row_idx = ui_graph.nonzero()[0]
            col_idx = ui_graph.nonzero()[1]
            keep_edge_idx = random.sample(list(range(len(ui_graph.data))), int(len(ui_graph.data) * (1 - self.drop_rate)))
            user_np = np.array(row_idx)[keep_edge_idx]
            item_np = np.array(col_idx)[keep_edge_idx]
            ratings = np.ones_like(user_np, dtype=np.float32)
            ui_graph_new = sp.csr_matrix((ratings, (user_np, item_np)), shape=(self.num_users, self.num_items))
            return gu_graph, ui_graph_new, gi_graph
            
            
        if aug_type == 5:#edge_drop gu
            row_idx = gu_graph.nonzero()[0]
            col_idx = gu_graph.nonzero()[1]
            keep_edge_idx = random.sample(list(range(len(gu_graph.data))), int(len(gu_graph.data) * (1 - self.drop_rate)))
            group_np = np.array(row_idx)[keep_edge_idx]
            user_np = np.array(col_idx)[keep_edge_idx]
            ratings = np.ones_like(user_np, dtype=np.float32)
            gu_graph_new = sp.csr_matrix((ratings, (group_np, user_np)), shape=(self.num_groups, self.num_users))
            return gu_graph_new, ui_graph, gi_graph         
            
        
    def gcn_propagate(self, graph, A_feature, B_feature):
        # node dropout on graph
        indices = graph._indices()
        values = graph._values()
        values = self.node_dropout(values)
        graph = torch.sparse.FloatTensor(
            indices, values, size=graph.shape)

        # propagate
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]
        for i in range(self.num_layers):
            features = self.mess_dropout(torch.matmul(graph, features))
            all_features.append(F.normalize(features))

        all_features = torch.stack(all_features, dim=1)
        all_features = torch.mean(all_features, dim=1)
        A_feature, B_feature = torch.split(
            all_features, (A_feature.shape[0], B_feature.shape[0]), 0)
        return A_feature, B_feature
        
    def two_propagate(self, graph, A_feature, B_feature, C_feature):
        # node dropout on graph
        indices = graph._indices()
        values = graph._values()
        values = self.node_dropout(values)
        graph = torch.sparse.FloatTensor(
            indices, values, size=graph.shape)
        # propagate
        features = torch.cat((A_feature, B_feature, C_feature), 0)
        all_features = [features]
        for i in range(self.num_layers):
            features = self.mess_dropout(torch.matmul(graph, features))
            all_features.append(F.normalize(features))
            
        temp_features = all_features

        all_features = torch.stack(all_features, dim = 1)
        all_features = torch.mean(all_features, dim = 1)
        A_feature, B_feature, C_feature= torch.split(
            all_features, (A_feature.shape[0], B_feature.shape[0], C_feature.shape[0]), 0)
        return A_feature, B_feature, C_feature, temp_features
    

    
    def cl_loss(self, global_user_embedding, local_user_embedding, global_item_embedding, local_item_embedding, user, item, type_m):
        
        user = user.reshape(-1)
        local_user_embedding_current = local_user_embedding[user]
        global_user_embedding_current = global_user_embedding[user]
        
        norm_user_emb1 = F.normalize(local_user_embedding_current)
        norm_user_emb2 = F.normalize(global_user_embedding_current)
        norm_all_user_emb = F.normalize(local_user_embedding)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.cl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.cl_temp).sum(axis=1)
        cl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()
        
        pos_item = item[:,0]
        
        local_item_embedding_current = local_item_embedding[pos_item]
        global_item_embedding_current = global_item_embedding[pos_item]
        
        
        norm_item_emb1 = F.normalize(local_item_embedding_current)
        norm_item_emb2 = F.normalize(global_item_embedding_current)
        norm_all_item_emb = F.normalize(local_item_embedding)
        
        
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.cl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.cl_temp).sum(dim=1)
        
        cl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()
        
        if type_m == 'g':
            cl_loss = self.cl_reg * (cl_loss_user + self.alpha * cl_loss_item)
        if type_m == 'u':
            cl_loss = self.cl_reg * cl_loss_user
        return cl_loss
    
    
    
    def cl_2_loss(self, context_embedding, center_embedding, user, item):
        
        _, old_item_embedding, old_user_embedding = torch.split(
            center_embedding, (self.num_users, self.num_items, self.num_groups),0)
        
        _, new_item_embedding, new_user_embedding = torch.split(
            context_embedding, (self.num_users, self.num_items, self.num_groups),0)
        
        user = user.reshape(-1)
        new_user_embedding_current = new_user_embedding[user]
        old_user_embedding_current = old_user_embedding[user]
        
        norm_user_emb1 = F.normalize(new_user_embedding_current)
        norm_user_emb2 = F.normalize(old_user_embedding_current)
        norm_all_user_emb = F.normalize(new_user_embedding)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.cl_2_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.cl_2_temp).sum(axis=1)
        cl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()
        
        pos_item = item[:,0]
        
        new_item_embedding_current = new_item_embedding[pos_item]
        old_item_embedding_current = old_item_embedding[pos_item]
        
        
        norm_item_emb1 = F.normalize(new_item_embedding_current)
        norm_item_emb2 = F.normalize(old_item_embedding_current)
        norm_all_item_emb = F.normalize(new_item_embedding)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.cl_2_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.cl_2_temp).sum(dim=1)
        
        cl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()
        
        cl_loss = self.cl_2_reg * (cl_loss_user + cl_loss_item)
        return cl_loss
        
        
    def propagate(self):
        
       
        whole_users_feature, whole_groups_feature, whole_items_feature, all_features = self.two_propagate(
            self.whole_graph_tensor, self.users_feature, self.groups_feature, self.items_feature)
        
        whole_users_feature_prime, whole_groups_feature_prime, whole_items_feature_prime,_ = self.two_propagate(
            self.whole_graph_tensor_prime, self.users_feature, self.groups_feature, self.items_feature)
        

        users_feature  = [whole_users_feature, whole_users_feature_prime]
        items_feature  = [whole_items_feature, whole_items_feature_prime]
        groups_feature = [whole_groups_feature, whole_groups_feature_prime]

        return users_feature, items_feature, groups_feature, all_features

    def evaluate(self, propagate_result, groups):
        users_feature, items_feature, groups_feature,_ = propagate_result
        groups_feature_atom, groups_feature_non_atom = [i[groups] for i in groups_feature] 
        items_feature_atom, items_feature_non_atom = items_feature 
        scores = torch.mm(groups_feature_atom, items_feature_atom.t()) #+ torch.mm(groups_feature_non_atom, items_feature_non_atom.t())
        return scores

    def predict(self, items_feature, groups_feature):
        items_feature_atom, items_feature_non_atom = items_feature 
        groups_feature_atom, groups_feature_non_atom = groups_feature
        pred = torch.sum(items_feature_atom * groups_feature_atom, 2)#+ torch.sum(items_feature_non_atom * groups_feature_non_atom, 2)
        return pred
    
    def forward(self, users, items, types):
        users_feature, items_feature, groups_feature, embeddings_list = self.propagate()
        items_embedding = [i[items] for i in items_feature] 
        context_embedding = embeddings_list[2]
        center_embedding = embeddings_list[0]
        
        if types == 'g':
            groups_embedding = [i[users] for i in groups_feature] 
            pred = self.predict(items_embedding, groups_embedding)
            L2_loss = self.regularize(items_embedding, groups_embedding)
            items_feature_atom, items_feature_non_atom = items_feature
            groups_feature_atom, groups_feature_non_atom = groups_feature
            cl_loss = self.cl_loss(groups_feature_atom, groups_feature_non_atom, items_feature_atom, items_feature_non_atom, users, items, 'g')
            cl_loss_2 = self.cl_2_loss(context_embedding,center_embedding, users,items)
            return pred, L2_loss, cl_loss, cl_loss_2
        if types == 'u':
            users_embedding = [i[users] for i in users_feature]
            L2_loss = self.regularize(items_embedding, users_embedding)
            items_feature_atom, items_feature_non_atom = items_feature
            users_feature_atom, users_feature_non_atom = users_feature
            cl_loss = self.cl_loss(users_feature_atom, users_feature_non_atom, items_feature_atom, items_feature_non_atom, users, items, 'u')
            return L2_loss, cl_loss
    

    def regularize(self, items_feature, groups_feature):
        items_feature_atom, items_feature_non_atom = items_feature 
        groups_feature_atom, groups_feature_non_atom = groups_feature  
        loss = self.embed_L2_norm * \
               ((items_feature_atom ** 2).sum() + (groups_feature_atom ** 2).sum())
        return loss
