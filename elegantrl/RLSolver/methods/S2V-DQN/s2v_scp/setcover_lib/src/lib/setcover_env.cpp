#include "setcover_env.h"
#include "graph.h"
#include <cassert>
#include <random>

SetCoverEnv::SetCoverEnv(double _norm) : IEnv(_norm)
{

}

void SetCoverEnv::s0(std::shared_ptr<Graph> _g)
{
    graph = _g;
    primal_set.clear();    
    dual_set.clear();
    action_list.clear();
    primal_edge_cnt.resize(graph->num_primal);
    for (int i = 0; i < graph->num_primal; ++i)
        primal_edge_cnt[i] = graph->adj_list[i].size();

    state_seq.clear();
    act_seq.clear();
    reward_seq.clear();
    sum_rewards.clear();
}

double SetCoverEnv::step(int a)
{
    assert(graph);
    assert(primal_set.count(a) == 0);

    state_seq.push_back(action_list);
    act_seq.push_back(a);

    primal_set.insert(a);
    action_list.push_back(a);

    for (auto& neigh : graph->adj_list[a])
        if (dual_set.count(neigh) == 0)
        {
            dual_set.insert(neigh);
            for (auto& p : graph->adj_list[neigh])
                primal_edge_cnt[p]--;
        }
    
    double r_t = getReward();
    reward_seq.push_back(r_t);
    sum_rewards.push_back(r_t);  

    return r_t;
}

int SetCoverEnv::randomAction()
{
    assert(graph);
    avail_list.clear();

    for (int i = 0; i < graph->num_primal; ++i)
        if (primal_set.count(i) == 0 && primal_edge_cnt[i])
        {
            avail_list.push_back(i);
        }
    
    assert(avail_list.size());
    int idx = rand() % avail_list.size();
    return avail_list[idx];
}

bool SetCoverEnv::isTerminal()
{
    assert(graph);
    return graph->num_dual == (int)dual_set.size();
}

double SetCoverEnv::getReward()
{
    return -1.0 / norm;
}