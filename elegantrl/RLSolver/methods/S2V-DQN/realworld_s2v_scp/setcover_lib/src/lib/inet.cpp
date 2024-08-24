#include "inet.h"

INet::INet()
{
    inputs.clear();
    param_record.clear();
    learner = new AdamOptimizer<mode, Dtype>(&model, cfg::learning_rate, cfg::l2_penalty);
}

void INet::UseOldModel()
{
    if (param_record.size() == 0)
    {
        for (auto& p : model.params)
        {
            param_record[p.first] = p.second->value.data;
        }
    }
    for (auto& p : model.params)
    {        
        assert(old_model.params.count(p.first));
        auto& old_ptr = old_model.params[p.first];
        p.second->SetRef(&(old_ptr->value));
    }
}

void INet::UseNewModel()
{
    assert(param_record.size());
    for (auto& p : param_record)
    {
        assert(model.params.count(p.first));
        auto& ptr = model.params[p.first];
        ptr->value.data = p.second;
    }
}