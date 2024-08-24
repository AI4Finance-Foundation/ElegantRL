#ifndef MVC_ENV_H
#define MVC_ENV_H

#include "i_env.h"

class MvcEnv : public IEnv
{
public:

    MvcEnv(double _norm);

    virtual void s0(std::shared_ptr<Graph>  _g) override;

    virtual double step(int a) override;

    virtual int randomAction() override;

    virtual bool isTerminal() override;

    virtual double getReward() override;

    int numCoveredEdges;
    std::set<int> covered_set;
    std::vector<int> avail_list;
};

#endif