#include "maxcut_env.h"
#include "graph.h"
#include <cassert>
#include <random>

MaxcutEnv::MaxcutEnv(double _norm) : IEnv(_norm)
{

}

void MaxcutEnv::s0(std::shared_ptr<Graph> _g)
{
    graph = _g;
    cut_set.clear();
    action_list.clear();
    cutWeight = 0;
    state_seq.clear();
    act_seq.clear();
    reward_seq.clear();
    sum_rewards.clear();
}

double MaxcutEnv::step(int a)
{
    assert(graph);
    assert(cut_set.count(a) == 0);
    assert(a >= 0 && a < graph->num_nodes);

    state_seq.push_back(action_list);
    act_seq.push_back(a);

    cut_set.insert(a);
    action_list.push_back(a);

    double old_cutWeight = cutWeight;
    for (auto& neigh : graph->adj_list[a])        
        if (cut_set.count(neigh.first) == 0)
            cutWeight += neigh.second;
        else
            cutWeight -= neigh.second;
    
    double r_t = getReward(old_cutWeight);
    reward_seq.push_back(r_t);
    sum_rewards.push_back(r_t);  

    return r_t;
}

int MaxcutEnv::randomAction()
{
    assert(graph);
    avail_list.clear();

    for (int i = 0; i < graph->num_nodes; ++i)
        if (cut_set.count(i) == 0)
            avail_list.push_back(i);
    
    assert(avail_list.size());
    int idx = rand() % avail_list.size();
    return avail_list[idx];
}

bool MaxcutEnv::isTerminal()
{
    assert(graph);
    return ((int)cut_set.size() + 1 >= graph->num_nodes);
}

double MaxcutEnv::getReward(double old_cutWeight)
{
    return (cutWeight - old_cutWeight) / norm;
}