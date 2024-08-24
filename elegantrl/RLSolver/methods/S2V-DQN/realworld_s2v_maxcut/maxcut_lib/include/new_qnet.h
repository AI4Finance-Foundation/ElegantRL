#ifndef NEW_Q_NET_H
#define NEW_Q_NET_H

#include "inet.h"
using namespace gnn;

class NewQNet : public INet
{
public:
    NewQNet();

    virtual void BuildNet() override;
    virtual void SetupTrain(std::vector<int>& idxes, 
                            std::vector< std::shared_ptr<Graph> >& g_list, 
                            std::vector< std::vector<int>* >& covered, 
                            std::vector<int>& actions, 
                            std::vector<double>& target) override;
                            
    virtual void SetupPredAll(std::vector<int>& idxes, 
                              std::vector< std::shared_ptr<Graph> >& g_list, 
                              std::vector< std::vector<int>* >& covered) override;

    void SetupGraphInput(std::vector<int>& idxes, 
                         std::vector< std::shared_ptr<Graph> >& g_list, 
                         std::vector< std::vector<int>* >& covered, 
                         const int* actions);

    SpTensor<CPU, Dtype> act_select, rep_global;
    SpTensor<mode, Dtype> m_act_select, m_rep_global;
};

#endif