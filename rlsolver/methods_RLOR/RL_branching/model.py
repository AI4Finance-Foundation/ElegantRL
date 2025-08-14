# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
# Contains model definitions for Siamese MLP and GNN architectures.             #
# Based on code by Labassi et al., 2022.                                        #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #

import torch as th
import torch.nn.functional as F
import torch_geometric


class MLPPolicy(th.nn.Module):

    def __init__(self, in_features=12):
        super().__init__()
        self.model = th.nn.Sequential(th.nn.Linear(in_features, 32),
                                      th.nn.LeakyReLU(),
                                      th.nn.Linear(32, 1))

    def forward(self, node1, node2):
        score1 = self.model(node1)
        score2 = self.model(node2)
        diff = -score1 + score2
        return th.sigmoid(diff)


class GNNPolicy(th.nn.Module):
    def __init__(self):
        super().__init__()

        emb_size = 32  # uniform node feature embedding dim

        hidden_dim1 = 8
        hidden_dim2 = 4
        hidden_dim3 = 4

        # static data
        cons_nfeats = 4
        edge_nfeats = 1
        var_nfeats = 6

        # CONSTRAINT EMBEDDING
        self.cons_embedding = th.nn.Sequential(
            th.nn.LayerNorm(cons_nfeats),
            th.nn.Linear(cons_nfeats, emb_size),
            th.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = th.nn.Sequential(
            th.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = th.nn.Sequential(
            th.nn.LayerNorm(var_nfeats),
            th.nn.Linear(var_nfeats, emb_size),
            th.nn.ReLU(),
        )

        self.bounds_embedding = th.nn.Sequential(
            th.nn.LayerNorm(2),
            th.nn.Linear(2, 2),
            th.nn.ReLU(),
        )

        # double check
        self.conv1 = torch_geometric.nn.GraphConv((emb_size, emb_size), hidden_dim1)
        self.conv2 = torch_geometric.nn.GraphConv((hidden_dim1, hidden_dim1), hidden_dim2)
        self.conv3 = torch_geometric.nn.GraphConv((hidden_dim2, hidden_dim2), hidden_dim3)

        self.convs = [self.conv1, self.conv2, self.conv3]

    def forward(self, batch, inv=False, epsilon=0.01):
        # create constraint masks. Constraints associated with variables
        # for which at least one of their bounds have changed
        # graph2 edges

        try:
            graph1 = (batch.constraint_features_s,
                      batch.edge_index_s,
                      batch.edge_attr_s,
                      batch.variable_features_s,
                      batch.bounds_s,
                      batch.constraint_features_s_batch,
                      batch.variable_features_s_batch)

            graph2 = (batch.constraint_features_t,
                      batch.edge_index_t,
                      batch.edge_attr_t,
                      batch.variable_features_t,
                      batch.bounds_t,
                      batch.constraint_features_t_batch,
                      batch.variable_features_t_batch)

        except AttributeError:
            graph1 = (batch.constraint_features_s,
                      batch.edge_index_s,
                      batch.edge_attr_s,
                      batch.variable_features_s,
                      batch.bounds_s)

            graph2 = (batch.constraint_features_t,
                      batch.edge_index_t,
                      batch.edge_attr_t,
                      batch.variable_features_t,
                      batch.bounds_t)

        if inv:
            graph1, graph2 = graph2, graph1

        # concatenation of averages variable/constraint features after conv
        score1 = self.forward_graph(*graph1)
        score2 = self.forward_graph(*graph2)
        return th.sigmoid(-score1 + score2)

    def forward_graph(self, constraint_features, edge_indices, edge_features,
                      variable_features, bbounds, constraint_batch=None, variable_batch=None):

        # Assume edge indices var to cons, constraint_mask of shape [Nconvs]
        variable_features = self.var_embedding(variable_features)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        bbounds = self.bounds_embedding(bbounds)

        edge_indices_reversed = th.stack([edge_indices[1], edge_indices[0]], dim=0)

        for conv in self.convs:
            # Var to cons
            constraint_features_next = F.relu(conv((variable_features, constraint_features),
                                                   edge_indices,
                                                   edge_weight=edge_features,
                                                   size=(variable_features.size(0), constraint_features.size(0))))

            # cons to var
            variable_features = F.relu(conv((constraint_features, variable_features),
                                            edge_indices_reversed,
                                            edge_weight=edge_features,
                                            size=(constraint_features.size(0), variable_features.size(0))))

            constraint_features = constraint_features_next

        if constraint_batch is not None:

            constraint_avg = torch_geometric.nn.pool.avg_pool_x(constraint_batch,
                                                                constraint_features,
                                                                constraint_batch)[0]
            variable_avg = torch_geometric.nn.pool.avg_pool_x(variable_batch,
                                                              variable_features,
                                                              variable_features)[0]
        else:
            constraint_avg = th.mean(constraint_features, dim=0, keepdim=True)
            variable_avg = th.mean(variable_features, dim=0, keepdim=True)

        return (th.linalg.norm(variable_avg, dim=1) +
                th.linalg.norm(constraint_avg, dim=1) +
                th.linalg.norm(bbounds, dim=1))
