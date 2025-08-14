import dgl
import torch
from torch import nn
from collections import OrderedDict
from itertools import combinations
from dgl.nn.pytorch.conv import GINConv, EGATConv

class Input_layer(nn.Module):
    def __init__(
            self,
            in_feat_row,
            in_feat_col,
            out_feat,
            weight_trans=True
    ):
        super().__init__()

        self.in_feat_row = in_feat_row
        self.in_feat_col = in_feat_col
        self.out_feat = out_feat
        self.weight_trans = weight_trans

        self.Linear_row = nn.Linear(self.in_feat_row, self.out_feat, bias=True)
        self.Linear_col = nn.Linear(self.in_feat_col, self.out_feat, bias=True)

    def forward(
            self,
            biGraph
    ):
        # node embedding update
        biGraph.nodes['row'].data['feat'] = self.Linear_row(biGraph.nodes['row'].data['feat'])
        biGraph.nodes['col'].data['feat'] = self.Linear_col(biGraph.nodes['col'].data['feat'])
        # edge weight update
        if self.weight_trans:
            biGraph.edges[('col', 'edge', 'row')].data['weight'] = torch.sigmoid(biGraph.edges[('col', 'edge', 'row')].data['weight'])
            biGraph.edges[('row', 'edge', 'col')].data['weight'] = torch.sigmoid(biGraph.edges[('row', 'edge', 'col')].data['weight'])
        return biGraph

class Conv_layer(nn.Module):
    def __init__(
            self,
            feat_node,
            use_residual=True,
            gin_mlp_layers=2,
            aggr_type='sum',
            learn_eps=True,
            activation=None
    ):
        super().__init__()

        self.feat_node = feat_node
        self.use_residual = use_residual
        self.gin_mlp_layers = gin_mlp_layers
        self.aggr_type = aggr_type
        self.learn_eps = learn_eps
        self.activation = activation

        list_c2r = []
        list_r2c = []
        for i in range(self.gin_mlp_layers):
            list_c2r.append(('Linear ' + str(i), nn.Linear(self.feat_node, self.feat_node)))
            list_r2c.append(('Linear ' + str(i), nn.Linear(self.feat_node, self.feat_node)))
            list_c2r.append(('LeakyReLU ' + str(i), nn.LeakyReLU()))
            list_r2c.append(('LeakyReLU ' + str(i), nn.LeakyReLU()))

        self.func_c2r = nn.Sequential(OrderedDict(list_c2r))
        self.func_r2c = nn.Sequential(OrderedDict(list_r2c))

        self.GINConv_c2r = GINConv(
            apply_func=self.func_c2r,
            aggregator_type=self.aggr_type,
            learn_eps=self.learn_eps,
            activation=self.activation
        )
        self.GINConv_r2c = GINConv(
            apply_func=self.func_r2c,
            aggregator_type=self.aggr_type,
            learn_eps=self.learn_eps,
            activation=self.activation
        )

    def forward(
            self,
            biGraph,
            edge_weight=True
    ):
        num_row = biGraph.num_nodes('row')
        num_col = biGraph.num_nodes('col')
        embedding_row = biGraph.nodes['row'].data['feat']
        embedding_col = biGraph.nodes['col'].data['feat']
        # col to row
        subGraph_c2r = biGraph.edge_type_subgraph([('col', 'edge', 'row')])
        homo_subGraph_c2r = dgl.to_homogeneous(subGraph_c2r, ndata=['feat'], edata=['weight'])
        if edge_weight:
            output = self.GINConv_c2r(homo_subGraph_c2r,
                                      homo_subGraph_c2r.ndata['feat'],
                                      homo_subGraph_c2r.edata['weight'])
        else:
            output = self.GINConv_c2r(homo_subGraph_c2r,
                                      homo_subGraph_c2r.ndata['feat'])
        biGraph.nodes['row'].data['feat'] = output[-num_row:, :]

        # row to col
        subGraph_r2c = biGraph.edge_type_subgraph([('row', 'edge', 'col')])
        homo_subGraph_r2c = dgl.to_homogeneous(subGraph_r2c, ndata=['feat'], edata=['weight'])
        if edge_weight:
            output = self.GINConv_r2c(homo_subGraph_r2c,
                                      homo_subGraph_r2c.ndata['feat'],
                                      homo_subGraph_r2c.edata['weight'])
        else:
            output = self.GINConv_r2c(homo_subGraph_r2c,
                                      homo_subGraph_r2c.ndata['feat'])
        biGraph.nodes['col'].data['feat'] = output[:num_col, :]

        if self.use_residual:
            biGraph.nodes['row'].data['feat'] += embedding_row
            biGraph.nodes['col'].data['feat'] += embedding_col

        return biGraph

class Graph_encoder(nn.Module):
    def __init__(
            self,
            in_feat_row=5,
            in_feat_col=11,
            mid_feat_node=128,
            num_conv_layers=3,
            use_residual=True,
            gin_mlp_layers=2,
            aggr_type='sum',
            learn_eps=True,
            activation=None
    ):
        super().__init__()

        self.in_feat_row = in_feat_row
        self.in_feat_col = in_feat_col
        self.mid_feat_node = mid_feat_node
        self.num_conv_layers = num_conv_layers
        self.use_residual = use_residual
        self.gin_mlp_layers = gin_mlp_layers
        self.aggr_type = aggr_type
        self.learn_eps = learn_eps
        self.activation = activation

        layer_list = [('Input 0',
                       Input_layer(
                           in_feat_row=self.in_feat_row,
                           in_feat_col=self.in_feat_col,
                           out_feat=self.mid_feat_node
                       ))]
        for i in range(self.num_conv_layers):
            layer_list.append(('Conv ' + str(i),
                               Conv_layer(
                                   feat_node=self.mid_feat_node,
                                   use_residual=self.use_residual,
                                   gin_mlp_layers=self.gin_mlp_layers,
                                   aggr_type=self.aggr_type,
                                   learn_eps=self.learn_eps,
                                   activation=self.activation
                               )))
        self.models = nn.Sequential(OrderedDict(layer_list))

    def forward(
            self,
            biGraph
    ):
        biGraph = self.models(biGraph)
        return biGraph

class Instance_encoder(nn.Module):
    def __init__(
            self,
            in_feat=4,
            out_feat=64,
            num_layer=3
    ):
        super().__init__()
        self.in_feat = in_feat
        self.mid_feat = out_feat
        self.out_feat = out_feat
        self.num_layer = num_layer

        mlp_list = []
        for i in range(self.num_layer):
            in_dim = self.mid_feat
            out_dim = self.mid_feat
            if i == 0:
                in_dim = self.in_feat
            if i == self.num_layer - 1:
                out_dim = out_feat
            mlp_list.append(('Linear ' + str(i), nn.Linear(in_dim, out_dim)))
            if i != self.num_layer - 1:
                mlp_list.append(('LeakyReLU ' + str(i), nn.LeakyReLU()))

        self.models = nn.Sequential(OrderedDict(mlp_list))

    def forward(
            self,
            global_feature
    ):
        output = self.models(global_feature)
        return output

class DQN_decoder(nn.Module):
    def __init__(
            self,
            feat_candidate=128,
            mid_feat=128,
            readout_layers=3
    ):
        super().__init__()
        self.feat_candidate = feat_candidate
        self.readout_layers = readout_layers
        self.mid_feat = mid_feat

        mlp_list = []
        for i in range(self.readout_layers):
            in_dim = self.mid_feat
            out_dim = self.mid_feat
            if i == 0:
                in_dim = self.feat_candidate
            if i == self.readout_layers - 1:
                out_dim = 1
            mlp_list.append(('Linear ' + str(i), nn.Linear(in_dim, out_dim)))
            if i != self.readout_layers - 1:
                mlp_list.append(('LeakyReLU ' + str(i), nn.LeakyReLU()))

        self.models = nn.Sequential(OrderedDict(mlp_list))
    def forward(
            self,
            batch,
            candidate_feature
    ):
        output = self.models(candidate_feature)  # (batch_size, num_candidate, 1)
        output = output.view(batch, -1)  # (batch_size, num_candidate)
        return output

class Critic_decoder(nn.Module):
    def __init__(
            self,
            feat_candidate=448,
            mid_feat=128,
            readout_layers=3
    ):
        super().__init__()
        self.feat_candidate = feat_candidate
        self.readout_layers = readout_layers
        self.mid_feat = mid_feat

        mlp_list = []
        for i in range(self.readout_layers):
            in_dim = self.mid_feat
            out_dim = self.mid_feat
            if i == 0:
                in_dim = self.feat_candidate  # [row_feature, column_feature, candidate_feature]
            if i == self.readout_layers - 1:
                out_dim = 1
            mlp_list.append(('Linear ' + str(i), nn.Linear(in_dim, out_dim)))
            if i != self.readout_layers - 1:
                mlp_list.append(('LeakyReLU ' + str(i), nn.LeakyReLU()))
        self.models = nn.Sequential(OrderedDict(mlp_list))

    def forward(
            self,
            feature
    ):
        output = self.models(feature)
        return output

class Actor_midlayer(nn.Module):  # Multiple column
    def __init__(
            self,
            feat_edge_in=2,
            feat_edge_mid=64,
            feat_candidate=128,
            num_heads=1,
            conv_layers=3,
            use_residual=True,
    ):
        super().__init__()
        self.feat_candidate = feat_candidate
        self.feat_edge_in = feat_edge_in
        self.feat_edge_mid = feat_edge_mid
        self.num_heads = num_heads
        self.conv_layers = conv_layers
        self.use_residual = use_residual

        self.input_edge_layer = nn.Linear(self.feat_edge_in, self.feat_edge_mid, bias=True)
        self.layer_list = nn.ModuleList([
            EGATConv(
                in_node_feats=self.feat_candidate,
                in_edge_feats=self.feat_edge_mid,
                out_node_feats=self.feat_candidate,
                out_edge_feats=self.feat_edge_mid,
                num_heads=1
            )
            for i in range(self.conv_layers)])

    def forward(
            self,
            fcGraph,
            node_feature,
            edge_feature
    ):
        edge_feature = self.input_edge_layer(edge_feature)

        for egat in self.layer_list:
            new_node_feature, new_edge_feature = egat(fcGraph, node_feature, edge_feature)
            new_node_feature = new_node_feature.squeeze()
            new_edge_feature = new_edge_feature.squeeze()
            if self.use_residual:
                node_feature = node_feature + new_node_feature
                edge_feature = edge_feature + new_edge_feature
            else:
                node_feature = new_node_feature
                edge_feature = new_edge_feature

        return node_feature

class Actor_decoder(nn.Module):  # Multiple column
    def __init__(
            self,
            feat_candidate=320,
            mid_feat=128,
            readout_layers=3,
            num_selections=5
    ):
        super().__init__()
        self.feat_candidate = feat_candidate
        self.mid_feat = mid_feat
        self.readout_layers = readout_layers
        self.num_selections = num_selections

        mlp_list = []
        for i in range(self.readout_layers):
            in_dim = self.mid_feat
            out_dim = self.mid_feat
            if i == 0:
                in_dim = self.feat_candidate
            mlp_list.append(('Linear ' + str(i), nn.Linear(in_dim, out_dim)))
            if i != self.readout_layers - 1:
                mlp_list.append(('LeakyReLU ' + str(i), nn.LeakyReLU()))

        read_list = [('W_o', nn.Linear(self.mid_feat, self.mid_feat, bias=False)),
                     ('ReLU', nn.ReLU()),
                     ('u', nn.Linear(self.mid_feat, 1, bias=False))]

        self.models = nn.Sequential(OrderedDict(mlp_list))
        self.read_function = nn.Sequential(OrderedDict(read_list))
        self.Softmax = nn.Softmax(dim=1)

    def forward(
            self,
            batch,
            candidate_feature
    ):
        output = self.models(candidate_feature)  # (batch_size, num_candidate, mid_feat)
        output = output.transpose(0, 1)  # (num_candidate, batch_size, mid_feat)
        output_comb = combinations(output, self.num_selections)
        multi_list = [sum(i) / self.num_selections for i in output_comb]
        multi_list = multi_list[: int(len(multi_list) / 2)]
        multi_embeddings = torch.stack(multi_list).transpose(0, 1)  # (batch_size, combinations, mid_feat)
        multi_logits = 10 * torch.tanh(self.read_function(multi_embeddings))
        multi_logits = multi_logits.squeeze(-1)
        output = self.Softmax(multi_logits)
        return output

class DQN_net(nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            num_candidate=10
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_candidate = num_candidate
        self.count = 0

    def forward(
            self,
            obs,
            state=None,
            info={}
    ):
        biGraph = obs.biGraph
        batch = len(biGraph)
        biGraph = dgl.batch(biGraph)
        biGraph = self.encoder(biGraph)
        biGraph = dgl.unbatch(biGraph)
        candidate_feature = [graph.nodes['col'].data['feat'][-self.num_candidate:] for graph in biGraph]
        candidate_feature = torch.stack(candidate_feature)  # (batch_size, num_candidate, feat_candidate)
        output = self.decoder(batch, candidate_feature)  # (batch_size, num_candidate)
        self.count += 1
        return output, state

class Critic_net(nn.Module):
    def __init__(
            self,
            graph_encoder,
            instance_encoder,
            critic_decoder,
            num_candidate=10
    ):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.instance_encoder = instance_encoder
        self.critic_decoder = critic_decoder
        self.num_candidate = num_candidate
        self.count = 0

    def forward(
            self,
            obs,
            state=None,
            info={}
    ):
        biGraph = obs.biGraph
        globalFeat = obs.globalFeat

        globalFeat = self.instance_encoder(globalFeat)  # (batch_size, feat_global)

        batch = len(biGraph)
        biGraph = dgl.batch(biGraph)
        biGraph = self.graph_encoder(biGraph)
        biGraph = dgl.unbatch(biGraph)
        row_feature = [torch.mean(graph.nodes['row'].data['feat'], dim=0) for graph in biGraph]
        column_feature = [torch.mean(graph.nodes['col'].data['feat'][:-self.num_candidate], dim=0) for graph in biGraph]
        candidate_feature = [torch.mean(graph.nodes['col'].data['feat'][-self.num_candidate:], dim=0) for graph in
                             biGraph]
        row_feature = torch.stack(row_feature)  # (batch_size, feat_candidate)
        column_feature = torch.stack(column_feature)
        candidate_feature = torch.stack(candidate_feature)
        feature = torch.cat([row_feature, column_feature, candidate_feature, globalFeat], dim=-1)  # (batch_size, 3 * feat_candidate + feat_global)

        output = self.critic_decoder(feature)  # (batch_size, 1)
        self.count += 1
        return output

class Actor_net(nn.Module):
    def __init__(
            self,
            graph_encoder,
            instance_encoder,
            actor_decoder,
            actor_midlayer,
            num_candidate=10,
            noise=0.0
    ):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.instance_encoder = instance_encoder
        self.actor_midlayer = actor_midlayer
        self.actor_decoder = actor_decoder
        self.num_candidate = num_candidate
        self.noise = noise
        self.count = 0

    def forward(
            self,
            obs,
            state=None,
            info={}
    ):
        biGraph = obs.biGraph
        fcGraph = obs.fcGraph
        globalFeat = obs.globalFeat

        batch = len(biGraph)
        biGraph = dgl.batch(biGraph)
        fcGraph = dgl.batch(fcGraph)

        biGraph = self.graph_encoder(biGraph)
        biGraph = dgl.unbatch(biGraph)

        candidate_feature = [graph.nodes['col'].data['feat'][-self.num_candidate:] for graph in biGraph]
        candidate_feature = torch.stack(candidate_feature)  # (batch_size, num_candidate, feat_candidate)
        mid_feature = candidate_feature

        shape = candidate_feature.shape  # (batch_size, num_candidate, feat_candidate)
        candidate_feature = candidate_feature.view(-1, shape[-1])   # (batch_size * num_candidate, feat_candidate)
        edge_feature = fcGraph.edata['feat']
        candidate_feature = self.actor_midlayer(fcGraph, candidate_feature, edge_feature)  # (batch_size * num_candidate, feat_candidate)
        candidate_feature = candidate_feature.view(shape)  # (batch_size, num_candidate, feat_candidate)

        globalFeat = self.instance_encoder(globalFeat)  # (batch_size, feat_global)
        globalFeat = torch.repeat_interleave(globalFeat, repeats=shape[1], dim=0)  # (batch_size, num_candidate * feat_global)
        globalFeat = globalFeat.view(shape[0], shape[1], globalFeat.shape[-1])  # (batch_size, num_candidate, feat_global)

        candidate_feature = torch.cat((mid_feature, candidate_feature, globalFeat), dim=-1)  # (batch_size, num_candidate, 2 * feat_candidate + feat_global)
        output = self.actor_decoder(batch, candidate_feature)  # (batch_size, num_candidate)
        output = (1 - self.noise) * output + self.noise * torch.ones_like(output) / output.shape[-1]
        self.count += 1
        return output, state