import torch as th
import torch.nn as nn

TEN = th.Tensor

'''graph_net'''


class GraphTRS(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, embed_dim, num_heads, num_layers):
        super().__init__()
        self.mlp_inp = BnMLP(dims=(inp_dim, inp_dim, mid_dim, embed_dim), activation=nn.GELU())

        '''trs_encoders + mlp'''
        self.trs_encoder_layers = []
        for layer_id in range(num_layers):
            trs_encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dropout=0., dim_feedforward=mid_dim, activation=nn.GELU())
            self.trs_encoder_layers.append(trs_encoder_layer)
            setattr(self, f'trs_encoder_layer{layer_id:02}', trs_encoder_layer)
        self.encoder_mlp = BnMLP(dims=(embed_dim, embed_dim, embed_dim), activation=nn.GELU())

        '''trs_decoders + mlp'''
        self.trs_decoder_layers = []
        for layer_id in range(num_layers):
            trs_decoder_layer = nn.TransformerDecoderLayer(
                d_model=embed_dim, nhead=num_heads, dropout=0., dim_feedforward=mid_dim, activation=nn.GELU())
            self.trs_decoder_layers.append(trs_decoder_layer)
            setattr(self, f'trs_decoder_layer{layer_id:02}', trs_decoder_layer)
        self.decoder_mlp = BnMLP(dims=(embed_dim, out_dim, out_dim), activation=None)

        num_nodes = inp_dim
        self.mlp_classifier = BnMLP(dims=(embed_dim, num_nodes, num_nodes), activation=nn.Softmax(dim=-1))

    def forward(self, inp0, mask):
        seq_input = self.mlp_inp(inp0)

        seq_encode1 = self.encoder_trs(src=seq_input, mask=mask)
        seq_encode2 = self.encoder_mlp(seq_encode1)

        seq_decode3 = self.decoder_trs(tgt=seq_encode2, memory=seq_encode1, mask=mask)
        seq_decode4 = self.decoder_mlp(seq_decode3)
        return seq_decode4, seq_encode2  # seq_output, seq_memory

    def encoder_trs(self, src, mask=None):
        # src: source tensor sequence
        for net in self.trs_encoder_layers:
            src = net(src=src, src_mask=mask)
        return src  # shape == (num_nodes, batch_size, embed_dim)

    def decoder_trs(self, tgt, memory, mask=None):
        # tgt: target tensor sequence
        # memory: memory tensor sequence
        for net in self.trs_decoder_layers:
            tgt = net(tgt=tgt, memory=memory, tgt_mask=mask, memory_mask=mask)
        return tgt  # shape == (num_nodes, batch_size, embed_dim)

    def get_seq_graph(self, seq_adj_float, mask):
        seq_input = self.mlp_inp(seq_adj_float)

        seq_encode1 = self.encoder_trs(src=seq_input, mask=mask)
        seq_encode2 = self.encoder_mlp(seq_encode1)
        return seq_encode2

    def get_node_classify(self, tgt):
        # tgt: target tensor sequence
        tgt = tgt / tgt.std(dim=-1, keepdim=True)
        return self.mlp_classifier(tgt)


def check_graph_net():
    # Example usage:
    num_nodes = 100
    num_heads = 4
    num_layers = 4
    inp_dim = num_nodes
    mid_dim = 256
    out_dim = num_nodes * 2
    embed_dim = int(inp_dim ** 0.5) - int(inp_dim ** 0.5) % num_heads
    batch_size = 3

    graph_net = GraphTRS(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim,
                         embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers)

    seq_len = num_nodes
    mask = create_mask(seq_len, mask_type='eye')
    seq_input = th.rand(seq_len, batch_size, inp_dim)

    seq_output, seq_memory = graph_net.forward(seq_input, mask)
    print(f"seq_input.shape  {seq_input.shape} == {(num_nodes, batch_size, inp_dim)}")
    print(f"seq_output.shape {seq_output.shape} == {(num_nodes, batch_size, out_dim)}")
    print(f"seq_memory.shape {seq_memory.shape} == {(num_nodes, batch_size, embed_dim)}")
    assert seq_input.shape == (num_nodes, batch_size, inp_dim)
    assert seq_output.shape == (num_nodes, batch_size, out_dim)
    assert seq_memory.shape == (num_nodes, batch_size, embed_dim)
    print()

    seq_graph = graph_net.get_seq_graph(seq_input, mask=mask)
    seq_classify = graph_net.get_node_classify(tgt=seq_graph)
    print(f"seq_graph.shape    {seq_graph.shape} == {(num_nodes, batch_size, embed_dim)}")
    print(f"seq_classify.shape {seq_classify.shape} == {(num_nodes, batch_size, num_nodes)}")
    assert seq_graph.shape == (num_nodes, batch_size, embed_dim)
    assert seq_classify.shape == (num_nodes, batch_size, num_nodes)
    print()


'''policy_net'''


class PolicyORG(nn.Module):
    def __init__(self, num_bits: int):
        super().__init__()
        self.num_nodes = num_bits
        self.out = nn.Parameter(th.rand((1, self.num_nodes)) * 0.02 + 0.49, requires_grad=True)

    def forward(self, xs_flt):
        assert isinstance(xs_flt, TEN)
        return th.sigmoid(self.out)

    def auto_regressive(self, xs_flt):
        return self.forward(xs_flt)


class PolicyMLP(nn.Module):
    def __init__(self, num_bits, mid_dim):
        super().__init__()
        self.net = BnMLP(dims=(num_bits, mid_dim, num_bits), activation=nn.Sigmoid())

    def auto_regressive(self, xs_flt):
        num_sims, num_nodes = xs_flt.shape
        device = xs_flt.device

        ids_ary = th.stack([th.randperm(num_nodes, device=device) for _ in range(num_sims)])
        sim_ids = th.arange(num_sims, device=device)

        probs = xs_flt.detach().clone()
        for i in range(num_nodes):
            ids = ids_ary[:, i]

            ps = self.net(probs.clone())[sim_ids, ids]
            probs[:, ids] = ps
        return probs


class PolicyRNN(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, embed_dim, num_heads, num_layers):
        super().__init__()
        assert num_heads > 0
        self.mlp_inp = BnMLP(dims=(embed_dim + 2, mid_dim, embed_dim), activation=nn.GELU())
        self.rnn = nn.GRU(embed_dim, mid_dim, num_layers=num_layers)
        self.mlp_out = BnMLP(dims=(embed_dim + mid_dim, mid_dim, out_dim), activation=nn.Sigmoid())

        self.mid_dim = mid_dim
        self.num_nodes = inp_dim
        self.embed_dim = embed_dim

    def forward(self, dec_node_i, xs_flt_i, prob, hidden):
        rnn_input = self.mlp_inp(th.concat((dec_node_i, xs_flt_i, prob), dim=1))
        rnn_output, hidden = self.rnn(rnn_input, hidden)
        prob = self.mlp_out(th.concat((dec_node_i, rnn_output), dim=1))
        return prob

    def auto_regressive(self, xs_flt, dec_node):
        num_nodes = dec_node.shape[0]
        device = xs_flt.device

        rnn_output = th.zeros((1, self.mid_dim), device=device)
        prob = self.mlp_out(th.concat((dec_node[0, :], rnn_output), dim=1))
        probs = [prob, ]
        hidden = None
        for i in range(1, num_nodes):
            prob = self.forward(dec_node[i, :], xs_flt[:, i, None], prob, hidden)
            probs.append(prob)
        return th.hstack(probs)


class PolicyTRS(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, embed_dim, num_heads, num_layers):
        super().__init__()
        self.mlp_inp = BnMLP(dims=(embed_dim + 1, mid_dim, embed_dim), activation=nn.GELU())

        self.trs_encoders = []
        for layer_id in range(num_layers):
            trs_encoder = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                     dropout=0., dim_feedforward=mid_dim, activation=nn.GELU())
            self.trs_encoders.append(trs_encoder)
            setattr(self, f'trs_encoder{layer_id:02}', trs_encoder)
        self.mlp_enc = BnMLP(dims=(embed_dim, embed_dim, embed_dim), activation=nn.GELU())

        self.mlp_node = BnMLP(dims=(embed_dim, mid_dim, mid_dim, mid_dim), activation=nn.GELU())
        self.mlp_prob = BnMLP(dims=(mid_dim + embed_dim, mid_dim, out_dim), activation=nn.Sigmoid())

        self.num_nodes = inp_dim

    def forward(self, x_flt, dec_node):
        # assert prob.shape == (num_nodes, )
        # assert dec_node.shape == (num_nodes, embed_dim)

        inp0 = th.concat((dec_node, x_flt[:, None]), dim=1)[:, None, :]
        # assert inp0.shape == (num_nodes, 1, embed_dim+1)
        enc0 = self.mlp_inp(inp0)
        # assert enc0.shape == (num_nodes, 1, embed_dim)

        enc1 = 0  # shape=(num_nodes, 1, embed_dim)
        for trs_encoder in self.trs_encoders:
            enc1 = enc1 + trs_encoder(enc0, src_mask=None)

        dec_prob = self.mlp_node(enc1)
        # assert dec_prob.shape == (num_nodes, 1, mid_dim)
        return dec_prob

    def auto_regressive(self, xs_flt, dec_node):
        # assert dec_node.shape == (num_nodes, batch_size, embed_dim)
        # assert dec_matrix.shape == (batch_size, mid_dim)
        i = 0
        dec_node = dec_node[:, i, :]
        dec_prob = self.forward(x_flt=xs_flt[i].clone().detach(), dec_node=dec_node)[:, i, :]

        dec = th.concat(tensors=(dec_node, dec_prob), dim=1)
        prob = self.mlp_prob(dec).squeeze(1)
        return prob[None, :]


def check_policy_net():
    import sys
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    '''config: graph'''
    num_nodes = 100

    '''config: model'''
    inp_dim = num_nodes
    out_dim = 1
    mid_dim = 128
    embed_dim = 32
    num_heads = 8
    num_layers = 4

    '''config: train'''
    num_sims = 3

    '''dummy data'''
    xs = th.rand((num_sims, num_nodes))
    dec_node = th.rand((num_nodes, 1, embed_dim), dtype=th.float32, device=device)

    '''model:MLP'''
    policy_net = PolicyMLP(num_bits=num_nodes, mid_dim=mid_dim).to(device)

    new_xs = policy_net.forward(xs_flt=xs.float())
    print('model:MLP', new_xs.shape)
    new_xs = policy_net.auto_regressive(xs_flt=xs.float())
    print('model:MLP', new_xs.shape)

    '''model:TRS'''
    policy_net = PolicyTRS(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim,
                           embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers).to(device)

    batch_idx = 0
    new_x = policy_net.forward(x_flt=xs[batch_idx].float(), dec_node=dec_node[:, batch_idx, :])
    print('model:TRS', new_x.shape)
    new_xs = policy_net.auto_regressive(xs_flt=xs.float(), dec_node=dec_node)
    print('model:TRS', new_xs.shape)


'''policy_trs_layer'''


class TrsDecoderLayer(nn.Module):
    def __init__(self, feature_dim: int, prob_dim: int, num_heads: int, mid_dim: int):
        super().__init__()
        self.f_dim = feature_dim
        self.p_dim = prob_dim
        embed_dim = feature_dim + prob_dim

        self.norm_f = nn.LayerNorm(feature_dim)
        self.norm_p = nn.LayerNorm(prob_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0)
        self.multi_head_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0)

        self.feed_forward = nn.Sequential(nn.LayerNorm(feature_dim),
                                          nn.Linear(feature_dim, mid_dim), nn.GELU(),
                                          nn.Linear(mid_dim, feature_dim), )

    def forward(self, seq_graph: TEN, seq_prob: TEN, seq_feature: TEN, tgt_mask=None, memory_mask=None):
        x = th.concat((seq_prob, seq_feature), dim=2)

        x_norm = th.concat((self.norm_p(seq_prob), self.norm_f(seq_feature)), dim=2)
        x = x + self.self_attn(x_norm, x_norm, x_norm, attn_mask=tgt_mask, need_weights=False)[0]

        memory = th.concat((self.norm_p(seq_prob), seq_graph.repeat(1, seq_prob.shape[1], 1)), dim=2)
        y = x + self.multi_head_attn(x, memory, memory, attn_mask=memory_mask, need_weights=False)[0]

        y_p, y_f = y.split((self.p_dim, self.f_dim), dim=2)
        y_f = y_f + self.feed_forward(y_f)
        return y_p, y_f


class PolicyTrsLayer(nn.Module):
    def __init__(self, mid_dim, feature_dim, prob_dim, num_heads, num_layers):
        super().__init__()
        prob_num = 2  # {True, False}

        '''TransformerDecoderLayer'''
        self.trs_decoder_layers = []
        for layer_id in range(num_layers):
            trs_decoder_layer = TrsDecoderLayer(
                feature_dim=feature_dim, prob_dim=prob_dim, mid_dim=mid_dim,
                num_heads=num_heads)
            self.trs_decoder_layers.append(trs_decoder_layer)
            setattr(self, f'trs_decoder_layer{layer_id:02}', trs_decoder_layer)

        '''probability dim reduction and extension'''
        self.prob_dim_reduce = nn.Sequential(nn.LayerNorm(prob_dim),
                                             nn.Linear(prob_dim, prob_dim), nn.GELU(),
                                             nn.Linear(prob_dim, prob_num), nn.Softmax(dim=-1))  # d => 2
        self.prob_dim_extend = nn.Sequential(nn.Linear(prob_num, prob_dim), nn.GELU(),
                                             nn.Linear(prob_dim, prob_dim), nn.GELU(), )  # 2 => d

    def forward(self, seq_graph, solution_xs, seq_feature, mask, layer_idx: int):
        x_prob0 = th.where(solution_xs.T, 3., -3.)[:, :, None]
        x_prob1 = -x_prob0
        seq_prob_2 = th.concat((x_prob0, x_prob1), dim=2)
        seq_prob_d = self.prob_dim_extend(seq_prob_2)
        # seq_prob_2.shape = (num_nodes, batch_size, 2)
        # seq_prob_d.shape = (num_nodes, batch_size, p_dim)

        trs_decoder_layer = self.trs_decoder_layers[layer_idx]
        seq_prob_d, seq_feature = trs_decoder_layer(seq_graph=seq_graph, seq_feature=seq_feature, seq_prob=seq_prob_d,
                                                    tgt_mask=mask, memory_mask=mask)

        seq_prob_2 = self.prob_dim_reduce(seq_prob_d)
        return seq_prob_2, seq_feature


def check_policy_trs_layer():
    import sys
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    '''config'''
    num_nodes = 100
    num_sims = 3
    embed_dim = 16
    num_heads = 8
    mid_dim = 256
    num_layers = 6

    seq_graph = th.rand((num_nodes, 1, embed_dim), dtype=th.float32, device=device)

    '''policy_net'''
    policy_net = PolicyTrsLayer(mid_dim=mid_dim, prob_dim=8, feature_dim=embed_dim,
                                num_heads=num_heads, num_layers=num_layers).to(device)
    mask = create_mask(seq_len=num_nodes, mask_type='eye')

    xs = th.randint(0, 2, size=(num_sims, num_nodes), dtype=th.bool, device=device)
    seq_feature = seq_graph.repeat(1, num_sims, 1)
    for layer_idx in range(num_layers):
        seq_prob, seq_feature = policy_net.forward(seq_graph=seq_graph, solution_xs=xs, seq_feature=seq_feature,
                                                   mask=mask, layer_idx=layer_idx)
        probs = seq_prob[:, :, 0].T
        print(f"layer_idx {layer_idx}  probs.shape {probs.shape}  seq_feature.shape {seq_feature.shape}")


'''utils'''


class BnMLP(nn.Module):
    def __init__(self, dims, activation=None):
        super(BnMLP, self).__init__()

        assert len(dims) >= 3
        mlp_list = [nn.Linear(dims[0], dims[1]), ]
        for i in range(1, len(dims) - 1):
            dim_i = dims[i]
            dim_j = dims[i + 1]
            mlp_list.extend([nn.GELU(), nn.LayerNorm(dim_i), nn.Linear(dim_i, dim_j)])

        if activation is not None:
            mlp_list.append(activation)

        self.mlp = nn.Sequential(*mlp_list)

        if activation is not None:
            layer_init_with_orthogonal(self.mlp[-2], std=0.1)
        else:
            layer_init_with_orthogonal(self.mlp[-1], std=0.1)

    def forward(self, inp):
        return self.mlp(inp)


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)


def create_mask(seq_len, mask_type):
    if mask_type == 'triu':
        # Create an upper triangular matrix with ones above the diagonal
        mask = th.triu(th.ones(seq_len, seq_len), diagonal=1)
    elif mask_type == 'eye':
        # Create a square matrix with zeros on the diagonal
        mask = th.eye(seq_len)
    else:
        raise ValueError("type should in ['triu', 'eye']")
    return mask.bool()  # True means not allowed to attend.


def reset_parameters_of_model(model: nn.Module):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


if __name__ == '__main__':
    check_graph_net()
    check_policy_trs_layer()
