#include "mvc_env.h"
#include "graph.h"
#include <cassert>
#include <random>

MvcEnv::MvcEnv(double _norm) : IEnv(_norm)
{

}

void MvcEnv::s0(std::shared_ptr<Graph> _g)
{
    graph = _g;
    covered_set.clear();
    action_list.clear();
    numCoveredEdges = 0;
    state_seq.clear();
    act_seq.clear();
    reward_seq.clear();
    sum_rewards.clear();
}

double MvcEnv::step(int a)
{
    assert(graph);
    assert(covered_set.count(a) == 0);

    state_seq.push_back(action_list);
    act_seq.push_back(a);

    covered_set.insert(a);
    action_list.push_back(a);

    for (auto& neigh : graph->adj_list[a])
        if (covered_set.count(neigh) == 0)
            numCoveredEdges++;
    
    double r_t = getReward();
    reward_seq.push_back(r_t);
    sum_rewards.push_back(r_t);  

    return r_t;
}

int MvcEnv::randomAction()
{
    assert(graph);
    avail_list.clear();

    for (int i = 0; i < graph->num_nodes; ++i)
        if (covered_set.count(i) == 0)
        {
            bool useful = false;
            for (auto& neigh : graph->adj_list[i])
                if (covered_set.count(neigh) == 0)
                {
                    useful = true;
                    break;
                }
            if (useful)
                avail_list.push_back(i);
        }
    
    assert(avail_list.size());
    int idx = rand() % avail_list.size();
    return avail_list[idx];
}

bool MvcEnv::isTerminal()
{
    assert(graph);
    return graph->num_edges == numCoveredEdges;
}

double MvcEnv::getReward()
{
    return -1.0 / norm;
}