#include "tsp2d_env.h"
#include "graph.h"
#include <cassert>
#include <random>
#include <iostream>

int sign = 1;

Tsp2dEnv::Tsp2dEnv(double _norm) : IEnv(_norm)
{

}

void Tsp2dEnv::s0(std::shared_ptr<Graph> _g)
{
    graph = _g;
    partial_set.clear();
    action_list.clear();
    action_list.push_back(0);
    partial_set.insert(0);

    state_seq.clear();
    act_seq.clear();
    reward_seq.clear();
    sum_rewards.clear();
}

double Tsp2dEnv::step(int a)
{
    assert(graph);
    assert(partial_set.count(a) == 0);
    assert(a > 0 && a < graph->num_nodes);

    state_seq.push_back(action_list);
    act_seq.push_back(a);

    double r_t = add_node(a);
    
    reward_seq.push_back(r_t);
    sum_rewards.push_back(r_t);  

    return r_t;
}

int Tsp2dEnv::randomAction()
{
    assert(graph);
    avail_list.clear();

    for (int i = 0; i < graph->num_nodes; ++i)
        if (partial_set.count(i) == 0)
            avail_list.push_back(i);
    
    assert(avail_list.size());
    int idx = rand() % avail_list.size();
    return avail_list[idx];
}

bool Tsp2dEnv::isTerminal()
{
    assert(graph);
    return ((int)action_list.size() == graph->num_nodes);
}

double Tsp2dEnv::add_node(int new_node)
{
    double cur_dist = 10000000.0;
    int pos = -1;
    for (size_t i = 0; i < action_list.size(); ++i)
    {
        int adj;
        if (i + 1 == action_list.size())
            adj = action_list[0];
        else
            adj = action_list[i + 1];
        double cost = graph->dist[new_node][action_list[i]]
                     + graph->dist[new_node][adj]
                     - graph->dist[action_list[i]][adj];
        if (cost < cur_dist)
        {
            cur_dist = cost;
            pos = i;
        }
    }
    assert(pos >= 0);
    assert(cur_dist >= -1e-8);
    action_list.insert(action_list.begin() + pos + 1, new_node);
    partial_set.insert(new_node);

    return sign * cur_dist / norm;
}