#include "simulator.h"
#include "graph.h"
#include "nstep_replay_mem.h"
#include "i_env.h"
#include "nn_api.h"
#include "config.h"

std::vector<IEnv*> Simulator::env_list;
std::vector< std::shared_ptr<Graph> > Simulator::g_list;
std::vector< std::vector<int>* > Simulator::covered;
std::vector< std::vector<double>* > Simulator::pred;

std::default_random_engine Simulator::generator;
std::uniform_real_distribution<double> Simulator::distribution(0.0,1.0);

void Simulator::Init(int num_env)
{
    env_list.resize(num_env);
    g_list.resize(num_env);
    covered.resize(num_env);
    pred.resize(num_env);
    for (int i = 0; i < num_env; ++i)
    {
        g_list[i] = nullptr;
        pred[i] = new std::vector<double>(cfg::max_n);
    }
}

void Simulator::run_simulator(int num_seq, double eps)
{
    int num_env = env_list.size();    
    int n = 0;
    while (n < num_seq)
    {
        for (int i = 0; i < num_env; ++i)
        {            
            if (!env_list[i]->graph || env_list[i]->isTerminal())
            {   
                if (env_list[i]->graph && env_list[i]->isTerminal())
                {
                    n++;
                    NStepReplayMem::Add(env_list[i]);
                }
                env_list[i]->s0(GSetTrain.Sample());
                g_list[i] = env_list[i]->graph;
                covered[i] = &(env_list[i]->action_list);
            }
        }

        if (n >= num_seq)
            break;            

        bool random = false;
        if (distribution(generator) >= eps)
            Predict(g_list, covered, pred);
        else        
            random = true;

        int a_t;
        for (int i = 0; i < num_env; ++i)
        {
            if (random)
                a_t = env_list[i]->randomAction();
            else 
                a_t = arg_max(env_list[i]->graph->num_nodes, pred[i]->data());
            env_list[i]->step(a_t);            
        }
    }
}

int arg_max(int n, const double* scores)
{
    int pos = -1; 
    double best = -10000000;
    for (int i = 0; i < n; ++i)
        if (pos == -1 || scores[i] > best)
        {
            pos = i;
            best = scores[i];
        }
    assert(pos >= 0);
    return pos;
}

double max(int n, const double* scores)
{
    int pos = -1; 
    double best = -10000000;
    for (int i = 0; i < n; ++i)
        if (pos == -1 || scores[i] > best)
        {
            pos = i;
            best = scores[i];
        }
    assert(pos >= 0);
    return best;
}
