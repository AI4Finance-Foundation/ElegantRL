import gzip
import pickle
import numpy as np
import torch as th
import torch_geometric
import torch.utils.data


class Dataset(th.utils.data.Dataset):
    def __init__(self, sample_files):
        super().__init__()
        self.sample_files = sample_files

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, index):
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        state1 = th.tensor(sample['state1'], dtype=th.float32)
        state2 = th.tensor(sample['state2'], dtype=th.float32)
        return (state1, state2), sample['action']


class BipartiteNodeData(torch_geometric.data.Data):
    def __init__(self, constraint_features, edge_index, edge_attr, variable_features, action):
        x = th.concatenate((constraint_features, variable_features))
        super().__init__(x, edge_index=edge_index, edge_attr=edge_attr)
        self.constraint_features = constraint_features
        self.variable_features = variable_features
        self.action = action

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return th.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


class BipartiteGraphPairData(torch_geometric.data.Data):
    """
    This class encode a pair of node bipartite graphs observation, s is graph0, t is graph1 
    """

    def __init__(self, constraint_features_s=None, edge_indices_s=None, edge_features_s=None,
                 variable_features_s=None, bounds_s=None, depth_s=None,
                 constraint_features_t=None, edge_indices_t=None, edge_features_t=None,
                 variable_features_t=None, bounds_t=None, depth_t=None,
                 action=None):

        super().__init__()

        self.variable_features_s = variable_features_s
        self.constraint_features_s = constraint_features_s
        self.edge_index_s = edge_indices_s
        self.edge_attr_s = edge_features_s
        self.bounds_s = bounds_s
        self.depth_s = depth_s

        self.variable_features_t = variable_features_t
        self.constraint_features_t = constraint_features_t
        self.edge_index_t = edge_indices_t
        self.edge_attr_t = edge_features_t
        self.bounds_t = bounds_t
        self.depth_t = depth_t

        self.action = action

    def __inc__(self, key, value, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs 
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == 'edge_index_s':
            return th.tensor([[self.variable_features_s.size(0)], [self.constraint_features_s.size(0)]])
        elif key == 'edge_index_t':
            return th.tensor([[self.variable_features_t.size(0)], [self.constraint_features_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """
    def __init__(self, sample_files):
        super().__init__()
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, idx):
        with gzip.open(self.sample_files[idx], 'rb') as f:
            sample = pickle.load(f)

        variable_features, (edge_indices, edge_features), constraint_features = sample['state']
        variable_features = th.FloatTensor(variable_features)
        constraint_features = th.FloatTensor(constraint_features)
        edge_indices = th.LongTensor(edge_indices.astype(np.int32))
        edge_features = th.FloatTensor(np.expand_dims(edge_features, axis=-1))

        return BipartiteNodeData(constraint_features, edge_indices, edge_features, variable_features, sample['action'])

    def torch_get(self, idx):
        data = th.load(self.sample_files[idx])
        return data
