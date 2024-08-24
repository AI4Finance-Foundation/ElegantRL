#ifndef SETCOVER_ENV_H
#define SETCOVER_ENV_H

#include "i_env.h"

class SetCoverEnv : public IEnv
{
public:

    SetCoverEnv(double _norm);

    virtual void s0(std::shared_ptr<Graph>  _g) override;

    virtual double step(int a) override;

    virtual int randomAction() override;

    virtual bool isTerminal() override;

    virtual double getReward() override;

    std::set<int> dual_set, primal_set;
    std::vector<int> avail_list;
    std::vector<int> primal_edge_cnt;
};

#endif