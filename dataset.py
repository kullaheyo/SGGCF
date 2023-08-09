import os
import torch
import numpy as np
import scipy.sparse as sp 
from configure import CONFIG
from torch.utils.data import Dataset


def sparse_ones(indices, size, dtype=torch.float):
    one = torch.ones(indices.shape[1], dtype=dtype)
    return torch.sparse.FloatTensor(indices, one, size=size).to(dtype)

def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), 
                                          torch.Size(graph.shape))
    return graph

def print_statistics(X, string):
    print('>'*10 + string + '>'*10 )
    print('Average interactions', X.sum(1).mean(0).item())
    nonzero_row_indice, nonzero_col_indice = X.nonzero()
    unique_nonzero_row_indice = np.unique(nonzero_row_indice)
    unique_nonzero_col_indice = np.unique(nonzero_col_indice)
    print('Non-zero rows', len(unique_nonzero_row_indice)/X.shape[0])
    print('Non-zero columns', len(unique_nonzero_col_indice)/X.shape[1])
    print('Matrix density', len(nonzero_row_indice)/(X.shape[0]*X.shape[1]))

class BasicDataset(Dataset):

    def __init__(self, path, name, task, neg_sample):
        self.path = path
        self.name = name
        self.task = task
        self.neg_sample = neg_sample
        self.num_users, self.num_items, self.num_groups = self.__load_data_size()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __load_data_size(self):
        with open(os.path.join(self.path, self.name, '{}_data_size.txt'.format(self.name)), 'r') as f:
            return [int(s) for s in f.readline().split('\t')][:3]
    def load_G_I_interaction(self):
        with open(os.path.join(self.path, self.name, 'group_item_{}.txt'.format(self.task)), 'r') as f:
            return list(map(lambda s: tuple(int(i) for i in s[:-1].split(',')), f.readlines()))
    def load_U_I_interaction(self):
        with open(os.path.join(self.path, self.name, 'user_item.txt'), 'r') as f:
            return list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
    def load_G_U_affiliation(self):
        with open(os.path.join(self.path, self.name, 'group_user.txt'), 'r') as f:
            return list(map(lambda s: tuple(int(float(i)) for i in s[:-1].split('\t')), f.readlines()))


class GroupTrainDataset(BasicDataset):
    def __init__(self, path, name):
        super().__init__(path, name, 'train', 1)
        # U-B
        self.G_I_pairs = self.load_G_I_interaction()
        indice = np.array(self.G_I_pairs, dtype=np.int32)
        values = np.ones(len(self.G_I_pairs), dtype=np.float32)
        self.ground_truth_g_i = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_groups, self.num_items)).tocsr()

        print_statistics(self.ground_truth_g_i, 'G_I statistics in train')


    def __getitem__(self, index):
        group_i, pos_item = self.G_I_pairs[index]
        all_items = [pos_item]
        if CONFIG['sample'] == 'simple':
            while True:
                i = np.random.randint(self.num_items)
                if self.ground_truth_g_i[group_i, i] == 0 and not i in all_items:
                    all_items.append(i)
                    if len(all_items) == self.neg_sample+1:
                        break
        else:
            raise ValueError(r"sample's method is wrong")

        return torch.LongTensor([group_i]), torch.LongTensor(all_items)

    def __len__(self):
        return len(self.G_I_pairs)


class GroupTestDataset(BasicDataset):
    def __init__(self, path, name, train_dataset, task='test'):
        super().__init__(path, name, task, None)
        # U-B
        self.G_I_pairs = self.load_G_I_interaction()
        indice = np.array(self.G_I_pairs, dtype=np.int32)
        values = np.ones(len(self.G_I_pairs), dtype=np.float32)
        self.ground_truth_g_i = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_groups, self.num_items)).tocsr()

        print_statistics(self.ground_truth_g_i, 'G-I statistics in test')

        self.train_mask_g_i = train_dataset.ground_truth_g_i
        self.groups = torch.arange(self.num_groups, dtype=torch.long)
        self.items = torch.arange(self.num_items, dtype=torch.long)
        assert self.train_mask_g_i.shape == self.ground_truth_g_i.shape

    def __getitem__(self, index):
        return index, torch.from_numpy(self.ground_truth_g_i[index].toarray()).squeeze(),  \
            torch.from_numpy(self.train_mask_g_i[index].toarray()).squeeze(),  \

    def __len__(self):
        return self.ground_truth_g_i.shape[0]


class UserDataset(BasicDataset):
    def __init__(self, path, name, assist_data, seed=None):
        super().__init__(path, name, 'train', 1)
        # U-I
        self.U_I_pairs = self.load_U_I_interaction()
        indice = np.array(self.U_I_pairs, dtype=np.int32)
        values = np.ones(len(self.U_I_pairs), dtype=np.float32)
        self.ground_truth_u_i = sp.coo_matrix( 
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_items)).tocsr()

        print_statistics(self.ground_truth_u_i, 'U-I statistics')

    def __getitem__(self, index):
        user_i, pos_item = self.U_I_pairs[index]
        all_items = [pos_item]
        while True:
            j = np.random.randint(self.num_items)
            if self.ground_truth_u_i[user_i, j] == 0 and not j in all_items:
                all_items.append(j)
                if len(all_items) == self.neg_sample+1:
                    break

        return torch.LongTensor([user_i]), torch.LongTensor(all_items)

    def __len__(self):
        return len(self.U_I_pairs)  


class AssistDataset(BasicDataset):
    def __init__(self, path, name):
        super().__init__(path, name, None, None)
        # B-I
        self.G_U_pairs = self.load_G_U_affiliation()
        indice = np.array(self.G_U_pairs, dtype=np.int32)
        values = np.ones(len(self.G_U_pairs), dtype=np.float32)
        self.ground_truth_g_u = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_groups, self.num_users)).tocsr()

        print_statistics(self.ground_truth_g_u,'G_U statistics')


def get_dataset(path, name, task='tune', seed=123):
    assist_data = AssistDataset(path, name)
    print('finish loading assist data')
    user_data = UserDataset(path, name, assist_data, seed=seed)
    print('finish loading item data')

    group_train_data = GroupTrainDataset(path, name)
    print('finish loading group train data')
    group_test_data = GroupTestDataset(path, name, group_train_data)
    print('finish loading group test data')

    return group_train_data, group_test_data, user_data, assist_data


