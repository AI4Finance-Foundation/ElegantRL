#include "config.h"
#include "tsp2d_lib.h"
#include "graph.h"
#include "nn_api.h"
#include "qnet.h"
#include "old_qnet.h"
#include "nstep_replay_mem.h"
#include "simulator.h"
#include "tsp2d_env.h"
#include <random>
#include <algorithm>
#include <cstdlib>
#include <signal.h>

using namespace gnn;
#define inf 2147483647/2

void intHandler(int dummy) {
    exit(0);
}

int LoadModel(const char* filename)
{
    ASSERT(net, "please init the lib before use");    
    net->model.Load(filename);
    return 0;
}

int SaveModel(const char* filename)
{
    ASSERT(net, "please init the lib before use");
    net->model.Save(filename);
    return 0;
}

std::vector< std::vector<double>* > list_pred;
Tsp2dEnv* test_env;
int Init(const int argc, const char** argv)
{
    signal(SIGINT, intHandler);
    
    cfg::LoadParams(argc, argv);
    GpuHandle::Init(cfg::dev_id, 1);

    if (!strcmp(cfg::net_type, "QNet"))
        net = new QNet();
    else if (!strcmp(cfg::net_type, "OldQNet"))
        net = new OldQNet();
    else {
        std::cerr << "unknown net type: " <<  cfg::net_type << std::endl;
        exit(0);
    }
    net->BuildNet();

    NStepReplayMem::Init(cfg::mem_size);
    
    Simulator::Init(cfg::num_env);
    for (int i = 0; i < cfg::num_env; ++i)
        Simulator::env_list[i] = new Tsp2dEnv(cfg::max_n);
    test_env = new Tsp2dEnv(cfg::max_n);

    list_pred.resize(cfg::batch_size);
    for (int i = 0; i < cfg::batch_size; ++i)
        list_pred[i] = new std::vector<double>(cfg::max_n + 10);
    return 0;
}

int UpdateSnapshot()
{
    net->old_model.DeepCopyFrom(net->model);
    return 0;
}

int InsertGraph(bool isTest, const int g_id, const int num_nodes, const double* coor_x, const double* coor_y)
{
    auto g = std::make_shared<Graph>(num_nodes, coor_x, coor_y);
    if (isTest)
        GSetTest.InsertGraph(g_id, g);
    else
        GSetTrain.InsertGraph(g_id, g);
    return 0;
}

int ClearTrainGraphs()
{
    GSetTrain.graph_pool.clear();
    return 0;
}

int PlayGame(const int n_traj, const double eps)
{
    Simulator::run_simulator(n_traj, eps);
    return 0;
}

ReplaySample sample;
std::vector<double> list_target;
double Fit(const double lr)
{
    NStepReplayMem::Sampling(cfg::batch_size, sample);
    bool ness = false;
    for (int i = 0; i < cfg::batch_size; ++i)
        if (!sample.list_term[i])
        {
            ness = true;
            break;
        }
    if (ness)
        PredictWithSnapshot(sample.g_list, sample.list_s_primes, list_pred);
    
    list_target.resize(cfg::batch_size);
    for (int i = 0; i < cfg::batch_size; ++i)
    {
        double q_rhs = 0;
        if (!sample.list_term[i])
            q_rhs = cfg::decay * max(sample.g_list[i]->num_nodes, list_pred[i]->data());
        q_rhs += sample.list_rt[i];
        list_target[i] = q_rhs;
    }

    return Fit(lr, sample.g_list, sample.list_st, sample.list_at, list_target);
}

double Tsp2dFarthest(std::shared_ptr<Graph> g, const std::vector<int>& s, const int act)
{
    std::vector<bool> used(g->num_nodes);
    for (int i = 0; i < g->num_nodes; ++i)
        used[i] = false;
    for (auto& x : s)
    {
	assert(!used[x]);
        used[x] = true;
    }

    std::vector<int> pos(g->num_nodes + 10), path_trace(g->num_nodes + 10);
    double cur_cost = 0;
    for (size_t i = 0; i < s.size(); ++i)
    {
        path_trace[i] = s[i];
	if (i)
		cur_cost += g->dist[s[i]][s[i-1]];
    }
    cur_cost += g->dist[s[0]][s[s.size() - 1]];
    path_trace[s.size()] = s[0];
    for (int t = s.size(); t < g->num_nodes; ++t) // insert 
    {
        double farthest = -1;
        int best_k = -1;
        // who to insert
        for (int i = 0; i < g->num_nodes; ++i)
            if (!used[i])
            {
                double best_dist = inf;
                for (int j = 0; j < t; ++j)
                {
                    double cost = g->dist[i][path_trace[j]];
                    if (cost < best_dist)
                        best_dist = cost;
                }
                if (best_dist < inf && best_dist > farthest)
                {
                    farthest = best_dist;
                    best_k = i;
                }
            }

        assert(best_k >= 0);            
	if (t == (int)s.size())
		best_k = act;
	assert(!used[best_k]);
        // where to insert
        double cur_dist = inf;
        for (int j = 0; j < t; ++j)
        {
            double cost = g->dist[best_k][path_trace[j]] + g->dist[best_k][path_trace[j + 1]] - g->dist[path_trace[j]][path_trace[j + 1]];
            if (cost < cur_dist)
            {
                cur_dist = cost;
                pos[best_k] = j;
            }
        }
        
        for (int p = t; p > pos[best_k]; --p)
            path_trace[p + 1] = path_trace[p];
        path_trace[pos[best_k] + 1] = best_k; 
        used[best_k] = true;
    }

    double best_sol = 0;
    for (int i = 0; i < g->num_nodes; ++i)
        best_sol += g->dist[path_trace[i]][path_trace[i + 1]];     
    return best_sol - cur_cost;
}

double FitWithFarthest(const double lr)
{
    NStepReplayMem::Sampling(cfg::batch_size, sample);
    list_target.resize(cfg::batch_size);
    for (int i = 0; i < cfg::batch_size; ++i)
    {
        list_target[i] = Tsp2dFarthest(sample.g_list[i], *(sample.list_st[i]), sample.list_at[i]);
    }
    return Fit(lr, sample.g_list, sample.list_st, sample.list_at, list_target);
}

double Test(const int gid)
{
    std::vector< std::shared_ptr<Graph> > g_list(1);
    std::vector< std::vector<int>* > states(1);

    test_env->s0(GSetTest.Get(gid));
    states[0] = &(test_env->action_list);
    g_list[0] = test_env->graph;

    double v = 0;
    int new_action;
    while (!test_env->isTerminal())
    {
        Predict(g_list, states, list_pred);
        auto& scores = *(list_pred[0]);
        new_action = arg_max(test_env->graph->num_nodes, scores.data());
        v += test_env->step(new_action) * cfg::max_n;
    }
    return v;
}

double GetSol(const int gid, int* sol)
{
    std::vector< std::shared_ptr<Graph> > g_list(1);
    std::vector< std::vector<int>* > states(1);

    test_env->s0(GSetTest.Get(gid));
    states[0] = &(test_env->action_list);
    g_list[0] = test_env->graph;

    double v = 0;
    int new_action;
    while (!test_env->isTerminal())
    {
        Predict(g_list, states, list_pred);
        auto& scores = *(list_pred[0]);
        new_action = arg_max(test_env->graph->num_nodes, scores.data());
        v += test_env->step(new_action) * cfg::max_n;
    }
    
    sol[0] = test_env->graph->num_nodes;
    for (int i = 0; i < test_env->graph->num_nodes; ++i)
        sol[i + 1] = test_env->action_list[i];    
    return v;
}

int SetSign(const int s)
{
    sign = s;
    return 0;
}

int ClearMem()
{
    NStepReplayMem::Clear();
    return 0;
}
