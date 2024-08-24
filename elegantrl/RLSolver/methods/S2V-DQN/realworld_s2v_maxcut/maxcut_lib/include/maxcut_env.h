#ifndef MAXCUT_ENV_H
#define MAXCUT_ENV_H

#include "i_env.h"

class MaxcutEnv : public IEnv
{
public:

    MaxcutEnv(double _norm);

    virtual void s0(std::shared_ptr<Graph>  _g) override;

    virtual double step(int a) override;

    virtual int randomAction() override;

    virtual bool isTerminal() override;

    double getReward(double old_cutWeight);

    double cutWeight;
    std::set<int> cut_set;
    std::vector<int> avail_list;
};

#endif