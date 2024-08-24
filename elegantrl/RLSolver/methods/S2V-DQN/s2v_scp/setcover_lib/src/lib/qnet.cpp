#include "qnet.h"
#include "graph.h"
#include "config.h"

QNet::QNet() : INet()
{
    inputs["node_feat"] = &m_node_feat;
    inputs["label"] = &m_y;
    inputs["graph"] = &graph;
    inputs["act_select"] = &m_act_select;
    inputs["rep_global"] = &m_rep_global;
    inputs["aux_feat"] = &m_aux_feat;
    cfg::aux_dim = 3;
    cfg::node_dim = 2;
}

void QNet::BuildNet()
{
    auto graph = add_const< GraphVar >(fg, "graph", true);    
    auto action_select = add_const< SpTensorVar<mode, Dtype> >(fg, "act_select", true);
    auto rep_global = add_const< SpTensorVar<mode, Dtype> >(fg, "rep_global", true);

	auto n2nsum_param = af< Node2NodeMsgPass<mode, Dtype> >(fg, {graph});
	auto subgsum_param = af< SubgraphMsgPass<mode, Dtype> >(fg, {graph});

	auto w_n2l = add_diff<DTensorVar>(model, "input-node-to-latent", {2, cfg::embed_dim});
	auto p_node_conv = add_diff< DTensorVar >(model, "linear-node-conv", {cfg::embed_dim, cfg::embed_dim});
    std::shared_ptr< DTensorVar<mode, Dtype> > h1_weight, h2_weight, last_w;

    if (cfg::reg_hidden > 0)
    {
        h1_weight = add_diff<DTensorVar>(model, "h1_weight", {2 * cfg::embed_dim, cfg::reg_hidden});
        h2_weight = add_diff<DTensorVar>(model, "h2_weight", {cfg::reg_hidden + cfg::aux_dim, 1});
        h2_weight->value.SetRandN(0, cfg::w_scale);
        fg.AddParam(h2_weight);
        last_w = h2_weight;
    } else 
    {
        h1_weight = add_diff<DTensorVar>(model, "h1_weight", {2 * cfg::embed_dim + cfg::aux_dim, 1});
        last_w = h1_weight;
    }

	w_n2l->value.SetRandN(0, cfg::w_scale);
	p_node_conv->value.SetRandN(0, cfg::w_scale);
	h1_weight->value.SetRandN(0, cfg::w_scale);
    fg.AddParam(w_n2l);
    fg.AddParam(p_node_conv);
    fg.AddParam(h1_weight);

	auto node_input = add_const< DTensorVar<mode, Dtype> >(fg, "node_feat", true);
	auto label = add_const< DTensorVar<mode, Dtype> >(fg, "label", true);

	auto input_message = af<MatMul>(fg, {node_input, w_n2l});
	auto input_potential_layer = af<ReLU>(fg, {input_message}); 
	int lv = 0;
	auto cur_message_layer = input_potential_layer;
	while (lv < cfg::max_bp_iter)
	{
		lv++;
		auto n2npool = af<MatMul>(fg, {n2nsum_param, cur_message_layer});
		auto node_linear = af<MatMul>(fg, {n2npool, p_node_conv});
		auto merged_linear = af<ElewiseAdd>(fg, {node_linear, input_message});
		cur_message_layer = af<ReLU>(fg, {merged_linear}); 
	}

	auto y_potential = af<MatMul>(fg, {subgsum_param, cur_message_layer});
    auto aux_input = add_const< DTensorVar<mode, Dtype> >(fg, "aux_feat", true);

    // Q func given a
    auto action_embed = af<MatMul>(fg, {action_select, cur_message_layer});
    auto embed_s_a = af< ConcatCols >(fg, {action_embed, y_potential});

    auto last_output = embed_s_a;
    if (cfg::reg_hidden > 0)
    {
        auto hidden = af<MatMul>(fg, {embed_s_a, h1_weight});
	    last_output = af<ReLU>(fg, {hidden}); 
    }
    last_output = af< ConcatCols >(fg, {last_output, aux_input});
    q_pred = af< MatMul >(fg, {last_output, last_w});

    auto diff = af< SquareError >(fg, {q_pred, label});
	loss = af< ReduceMean >(fg, {diff});

    // q func on all a
    auto rep_y = af<MatMul>(fg, {rep_global, y_potential});
    auto embed_s_a_all = af< ConcatCols >(fg, {cur_message_layer, rep_y});

    last_output = embed_s_a_all;
    if (cfg::reg_hidden > 0)
    {
        auto hidden = af<MatMul>(fg, {embed_s_a_all, h1_weight});
	    last_output = af<ReLU>(fg, {hidden}); 
    }

    auto rep_aux = af<MatMul>(fg, {rep_global, aux_input});
    last_output = af< ConcatCols >(fg, {last_output, rep_aux});
    q_on_all = af< MatMul >(fg, {last_output, last_w});
}

int QNet::GetStatusInfo(std::shared_ptr<Graph> g, int num, const int* covered, int& counter, std::vector<int>& idx_map)
{
    std::set<int> c;
    idx_map.resize(g->num_nodes);
    for (int i = 0; i < g->num_nodes; ++i)
        idx_map[i] = -1;

    for (int i = 0; i < num; ++i)
    {
        auto p = covered[i];
        assert(g->is_primal(p));
        for (auto& x : g->adj_list[p])
            c.insert(x);
    }       
    counter = 0; 
    int n = 0;

    for (int i = g->num_primal; i < g->num_nodes; ++i)
        if (!c.count(i))
        {
            idx_map[i] = 0;
            n++;
        }

    for (int i = 0; i < g->num_primal; ++i)
    {
        bool useful = false;
        for (auto& x : g->adj_list[i])
            if (!c.count(x))
            {
                useful = true;
                break;
            }
        if (useful)
        {
            idx_map[i] = 0;
            n++;
        }            
    }
    counter = c.size();
    return n;
}

void QNet::SetupGraphInput(std::vector<int>& idxes, 
                           std::vector< std::shared_ptr<Graph> >& g_list, 
                           std::vector< std::vector<int>* >& covered, 
                           const int* actions)
{
    idx_map_list.resize(idxes.size());
    avail_act_cnt.resize(idxes.size());
    aux_feat.Reshape({idxes.size(), (size_t)cfg::aux_dim});
    aux_feat.Fill(0.0);

	int node_cnt = 0;
    for (size_t i = 0; i < idxes.size(); ++i)
    {
        auto& g = g_list[idxes[i]];
        int counter;
        auto* aux_ptr = aux_feat.data->ptr + cfg::aux_dim * i;
        if (g->num_primal)
            aux_ptr[0] = (Dtype)covered[idxes[i]]->size() / (Dtype)g->num_primal;

        avail_act_cnt[i] = GetStatusInfo(g, covered[idxes[i]]->size(), covered[idxes[i]]->data(), counter, idx_map_list[i]);
        if (g->num_dual)
            aux_ptr[1] = (Dtype)counter / (Dtype)g->num_dual;

        aux_ptr[2] = 1.0;
        node_cnt += avail_act_cnt[i];
    }
    graph.Resize(idxes.size(), node_cnt);
    node_feat.Reshape({(size_t)node_cnt, (size_t)cfg::node_dim});
    node_feat.Fill(1.0);

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
    int edge_cnt = 0;
    for (size_t i = 0; i < idxes.size(); ++i)
	{                
        auto& g = g_list[idxes[i]];
        auto& idx_map = idx_map_list[i];

        int t = 0;
        for (int j = 0; j < g->num_nodes; ++j)
        {
            if (idx_map[j] < 0)
                continue;
            idx_map[j] = t;
            graph.AddNode(i, node_cnt + t);
            if (!actions)
            {
                rep_global.data->row_ptr[node_cnt + t] = node_cnt + t;
                rep_global.data->val[node_cnt + t] = 1.0;
                rep_global.data->col_idx[node_cnt + t] = i;
            }
            if (g->is_primal(j))
            {
                node_feat.data->ptr[cfg::node_dim * (node_cnt + t)] = 0;
            }
            t += 1;
        }
        assert(t == avail_act_cnt[i]);
        
        if (actions)
        {
            auto act = actions[idxes[i]];
            assert(idx_map[act] >= 0 && act >= 0 && act < g->num_nodes);
            act_select.data->row_ptr[i] = i;
            act_select.data->val[i] = 1.0;
            act_select.data->col_idx[i] = node_cnt + idx_map[act];
        }

        for (int j = 0; j < g->num_nodes; ++j)
            if (idx_map[j] >= 0)
            {
                for (auto& x: g->adj_list[j])
                    if (idx_map[x] >= 0)
                    {
                        assert(g->is_primal(j) ^ g->is_primal(x));
                        graph.AddEdge(edge_cnt, idx_map[j] + node_cnt, idx_map[x] + node_cnt);
                        edge_cnt++;
                    }
            }

        node_cnt += avail_act_cnt[i];
	}
    assert(node_cnt == (int)graph.num_nodes);
    if (actions)
    {
        act_select.data->row_ptr[idxes.size()] = idxes.size();
        m_act_select.CopyFrom(act_select);
    } else {
        rep_global.data->row_ptr[node_cnt] = node_cnt;
        m_rep_global.CopyFrom(rep_global);
    }

    m_aux_feat.CopyFrom(aux_feat);
    m_node_feat.CopyFrom(node_feat);
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