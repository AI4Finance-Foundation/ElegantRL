#include "qnet.h"
#include "graph.h"
#include "config.h"

QNet::QNet() : INet()
{
    inputs["node_feat"] = &m_node_feat;
    inputs["edge_feat"] = &m_edge_feat;
    inputs["label"] = &m_y;
    inputs["graph"] = &graph;
    inputs["act_select"] = &m_act_select;
    inputs["rep_global"] = &m_rep_global;
    cfg::node_dim = 2;
    cfg::edge_dim = 4;
}

void QNet::BuildNet()
{
    auto graph = add_const< GraphVar >(fg, "graph", true);    
    auto action_select = add_const< SpTensorVar<mode, Dtype> >(fg, "act_select", true);
    auto rep_global = add_const< SpTensorVar<mode, Dtype> >(fg, "rep_global", true);

	auto n2nsum_param = af< Node2NodeMsgPass<mode, Dtype> >(fg, {graph});
    auto e2nsum_param = af< Edge2NodeMsgPass<mode, Dtype> >(fg, {graph});
	auto subgsum_param = af< SubgraphMsgPass<mode, Dtype> >(fg, {graph}, cfg::avg_global);

	auto w_n2l = add_diff<DTensorVar>(model, "input-node-to-latent", {2, cfg::embed_dim});
    auto w_e2l = add_diff<DTensorVar>(model, "input-edge-to-latent", {4, cfg::edge_embed_dim});
	auto p_node_conv = add_diff< DTensorVar >(model, "linear-node-conv", {cfg::embed_dim * 2 + cfg::edge_embed_dim, cfg::embed_dim});
    std::shared_ptr< DTensorVar<mode, Dtype> > h1_weight, h2_weight, last_w;

    if (cfg::reg_hidden > 0)
    {
        h1_weight = add_diff<DTensorVar>(model, "h1_weight", {2 * cfg::embed_dim, cfg::reg_hidden});
        h2_weight = add_diff<DTensorVar>(model, "h2_weight", {cfg::reg_hidden, 1});
        h2_weight->value.SetRandN(0, cfg::w_scale);
        fg.AddParam(h2_weight);
        last_w = h2_weight;
    } else 
    {
        h1_weight = add_diff<DTensorVar>(model, "h1_weight", {2 * cfg::embed_dim, 1});
        last_w = h1_weight;
    }

	w_n2l->value.SetRandN(0, cfg::w_scale);
    w_e2l->value.SetRandN(0, cfg::w_scale);
	p_node_conv->value.SetRandN(0, cfg::w_scale);
	h1_weight->value.SetRandN(0, cfg::w_scale);
    fg.AddParam(w_n2l);
    fg.AddParam(w_e2l);
    fg.AddParam(p_node_conv);
    fg.AddParam(h1_weight);

	auto node_input = add_const< DTensorVar<mode, Dtype> >(fg, "node_feat", true);
    auto edge_input = add_const< DTensorVar<mode, Dtype> >(fg, "edge_feat", true);
	auto label = add_const< DTensorVar<mode, Dtype> >(fg, "label", true);

    auto node_init = af<MatMul>(fg, {node_input, w_n2l});
    node_init = af<ReLU>(fg, {node_init});
    auto edge_init = af< MatMul >(fg, {edge_input, w_e2l});
    edge_init = af<ReLU>(fg, {edge_init});
    auto e2npool = af<MatMul>(fg, {e2nsum_param, edge_init});

	int lv = 0;
	auto cur_node_embed = node_init;
	while (lv < cfg::max_bp_iter)
	{
		lv++;
		auto n2npool = af<MatMul>(fg, {n2nsum_param, cur_node_embed});
        auto node_concat = af< ConcatCols >(fg, {n2npool, cur_node_embed, e2npool});

		auto node_linear = af<MatMul>(fg, {node_concat, p_node_conv});
		cur_node_embed = af<ReLU>(fg, {node_linear}); 
	}

	auto y_potential = af<MatMul>(fg, {subgsum_param, cur_node_embed});

    // Q func given a
    auto action_embed = af<MatMul>(fg, {action_select, cur_node_embed});
    auto embed_s_a = af< ConcatCols >(fg, {action_embed, y_potential});

    auto last_output = embed_s_a;
    if (cfg::reg_hidden > 0)
    {
        auto hidden = af<MatMul>(fg, {embed_s_a, h1_weight});
	    last_output = af<ReLU>(fg, {hidden}); 
    }
    q_pred = af< MatMul >(fg, {last_output, last_w});

    auto diff = af< SquareError >(fg, {q_pred, label});
	loss = af< ReduceMean >(fg, {diff});

    // q func on all a
    auto rep_y = af<MatMul>(fg, {rep_global, y_potential});
    auto embed_s_a_all = af< ConcatCols >(fg, {cur_node_embed, rep_y});

    last_output = embed_s_a_all;
    if (cfg::reg_hidden > 0)
    {
        auto hidden = af<MatMul>(fg, {embed_s_a_all, h1_weight});
	    last_output = af<ReLU>(fg, {hidden}); 
    }

    q_on_all = af< MatMul >(fg, {last_output, last_w});    
}

void QNet::SetupGraphInput(std::vector<int>& idxes, 
                           std::vector< std::shared_ptr<Graph> >& g_list, 
                           std::vector< std::vector<int>* >& covered, 
                           const int* actions)
{
	int node_cnt = 0, edge_cnt = 0;
    for (size_t i = 0; i < idxes.size(); ++i)
    {
        auto& g = g_list[idxes[i]];
        node_cnt += g->num_nodes;
        edge_cnt += g->num_edges * 2;
    }
    graph.Resize(idxes.size(), node_cnt);
    node_feat.Reshape({(size_t)node_cnt, (size_t)cfg::node_dim});
    node_feat.Fill(1.0);
    edge_feat.Reshape({(size_t)edge_cnt, (size_t)cfg::edge_dim});
    edge_feat.Fill(1.0);

    if (actions)
    {
        act_select.Reshape({idxes.size(), (size_t)node_cnt});
        act_select.ResizeSp(idxes.size(), idxes.size() + 1);
    } else
    {
        rep_global.Reshape({(size_t)node_cnt, idxes.size()});
        rep_global.ResizeSp(node_cnt, node_cnt + 1);
    }
    node_cnt = 0;
    edge_cnt = 0;
    size_t edge_offset = 0;
    for (size_t i = 0; i < idxes.size(); ++i)
	{                
        auto& g = g_list[idxes[i]];
        std::set<int> c;
        for (size_t j = 0; j < covered[idxes[i]]->size(); ++j)
        {
            auto& cc = *(covered[idxes[i]]);
            int n_c = cc[j];
            c.insert(n_c);
            node_feat.data->ptr[2 * (node_cnt + n_c)] = 0.0;
        }
            
        for (int j = 0; j < g->num_nodes; ++j)
        {
            int x = node_cnt + j;
            graph.AddNode(i, x);
            for (auto& p : g->adj_list[j])
            {
                graph.AddEdge(edge_cnt, x, node_cnt + p.first);
                auto* edge_ptr = edge_feat.data->ptr + edge_offset;
                edge_ptr[0] = c.count(j);
                edge_ptr[1] = p.second;
                edge_ptr[2] = c.count(p.first) ^ c.count(j);

                edge_offset += cfg::edge_dim;
                edge_cnt++;
            }
            if (!actions)
            {
                rep_global.data->row_ptr[node_cnt + j] = node_cnt + j;
                rep_global.data->val[node_cnt + j] = 1.0;
                rep_global.data->col_idx[node_cnt + j] = i;
            }
        }
        if (actions)
        {
            auto act = actions[idxes[i]];
            assert(act >= 0 && act < g->num_nodes);
            act_select.data->row_ptr[i] = i;
            act_select.data->val[i] = 1.0;
            act_select.data->col_idx[i] = node_cnt + act;
        }
        node_cnt += g->num_nodes;
	}
    assert(edge_offset == edge_feat.shape.Count());
    assert(edge_cnt == (int)graph.num_edges);
    assert(node_cnt == (int)graph.num_nodes);
    if (actions)
    {
        act_select.data->row_ptr[idxes.size()] = idxes.size();
        m_act_select.CopyFrom(act_select);
    } else {
        rep_global.data->row_ptr[node_cnt] = node_cnt;
        m_rep_global.CopyFrom(rep_global);
    }

    m_node_feat.CopyFrom(node_feat);
    m_edge_feat.CopyFrom(edge_feat);
}

void QNet::SetupTrain(std::vector<int>& idxes, 
                      std::vector< std::shared_ptr<Graph> >& g_list, 
                      std::vector< std::vector<int>* >& covered, 
                      std::vector<int>& actions, 
                      std::vector<double>& target)
{    
    SetupGraphInput(idxes, g_list, covered, actions.data());

    y.Reshape({idxes.size(), (size_t)1});
    for (size_t i = 0; i < idxes.size(); ++i)
        y.data->ptr[i] = target[idxes[i]];
    m_y.CopyFrom(y);
}

void QNet::SetupPredAll(std::vector<int>& idxes, 
                        std::vector< std::shared_ptr<Graph> >& g_list, 
                        std::vector< std::vector<int>* >& covered)
{    
    SetupGraphInput(idxes, g_list, covered, nullptr);
}