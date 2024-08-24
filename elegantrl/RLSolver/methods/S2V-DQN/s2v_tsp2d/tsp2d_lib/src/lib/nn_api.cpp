#include "config.h"
#include "nn_api.h"
#include "graph.h"
#include "tensor/tensor_all.h"
#include <random>
#include <algorithm>
#include <cstdlib>
#include <signal.h>
using namespace gnn;

#define inf 2147483647/2

INet* net = nullptr;

std::vector<int> batch_idxes;

void Predict(std::vector< std::shared_ptr<Graph> >& g_list, std::vector< std::vector<int>* >& covered, std::vector< std::vector<double>* >& pred)
{
    DTensor<CPU, Dtype> output;
    int n_graphs = g_list.size();
    for (int i = 0; i < n_graphs; i += cfg::batch_size)
    {
        int bsize = cfg::batch_size;
        if (i + cfg::batch_size > n_graphs)
            bsize = n_graphs - i;
        batch_idxes.resize(bsize);
        for (int j = i; j < i + bsize; ++j)
            batch_idxes[j - i] = j;
        
        net->SetupPredAll(batch_idxes, g_list, covered);
        net->fg.FeedForward({net->q_on_all}, net->inputs, Phase::TEST);
        auto& raw_output = net->q_on_all->value;
        output.CopyFrom(raw_output);
        
        int pos = 0;
        for (int j = i; j < i + bsize; ++j)
        {
            auto& cur_pred = *(pred[j]);
            auto& g = g_list[j];

            for (int k = 0; k < g->num_nodes; ++k)
            {
                cur_pred[k] = output.data->ptr[pos];
                pos += 1;
            }            
            auto& cur_covered = *(covered[j]);
            for (auto& k : cur_covered)
                cur_pred[k] = -inf;
        }
        ASSERT(pos == (int)output.shape.Count(), "idxes not match");
    }   
}

void PredictWithSnapshot(std::vector< std::shared_ptr<Graph> >& g_list, std::vector< std::vector<int>* >& covered, std::vector< std::vector<double>* >& pred)
{
    net->UseOldModel();
    Predict(g_list, covered, pred);
    net->UseNewModel();
}

double Fit(const double lr, std::vector< std::shared_ptr<Graph> >& g_list, std::vector< std::vector<int>* >& covered, std::vector<int>& actions, std::vector<double>& target)
{   
    Dtype loss = 0;
    int n_graphs = g_list.size();
    for (int i = 0; i < n_graphs; i += cfg::batch_size)
    {
        int bsize = cfg::batch_size;
        if (i + cfg::batch_size > n_graphs)
            bsize = n_graphs - i;

        batch_idxes.resize(bsize);
        for (int j = i; j < i + bsize; ++j)
            batch_idxes[j - i] = j;

        net->SetupTrain(batch_idxes, g_list, covered, actions, target);
        net->fg.FeedForward({net->loss}, net->inputs, Phase::TRAIN);
        net->fg.BackPropagate({net->loss});
        net->learner->cur_lr = lr;
        net->learner->Update();

        loss += net->loss->AsScalar() * bsize;
    }
    
    return loss / g_list.size();
}
