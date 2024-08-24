#ifndef INET_H
#define INET_H

#include <map>
#include <string>
#include <vector>
#include "config.h"
#include "tensor/tensor.h"
#include "nn/nn_all.h"
#include "util/graph_struct.h"

#include "graph.h"

using namespace gnn;

class INet
{
public:
    INet();

    virtual void BuildNet() = 0;

    virtual void SetupTrain(std::vector<int>& idxes, 
                            std::vector< std::shared_ptr<Graph> >& g_list, 
                            std::vector< std::vector<int>* >& covered, 
                            std::vector<int>& actions, 
                            std::vector<double>& target) = 0;
                            
    virtual void SetupPredAll(std::vector<int>& idxes, 
                              std::vector< std::shared_ptr<Graph> >& g_list, 
                              std::vector< std::vector<int>* >& covered) = 0;

    void UseOldModel();
    void UseNewModel();
    
    DTensor<CPU, Dtype> node_feat, edge_feat, y;
    DTensor<mode, Dtype> m_node_feat, m_edge_feat, m_y;
    GraphStruct graph;
    FactorGraph fg;
    ParamSet<mode, Dtype> model, old_model;
    AdamOptimizer<mode, Dtype>* learner;

    std::map< std::string, void* > inputs;
    std::map<std::string, std::shared_ptr< DenseData<mode, Dtype> > > param_record;
    std::shared_ptr< DTensorVar<mode, Dtype> > loss, q_pred, q_on_all;
};

#endif