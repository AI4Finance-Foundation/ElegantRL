#ifndef NN_API_H
#define NN_API_H

#include "inet.h"

extern INet* net;

void Predict(std::vector< std::shared_ptr<Graph> >& g_list, std::vector< std::vector<int>* >& covered, std::vector< std::vector<double>* >& pred);

void PredictWithSnapshot(std::vector< std::shared_ptr<Graph> >& g_list, std::vector< std::vector<int>* >& covered, std::vector< std::vector<double>* >& pred);

double Fit(std::vector< std::shared_ptr<Graph> >& g_list, std::vector< std::vector<int>* >& covered, std::vector<int>& actions, std::vector<double>& target);

#endif