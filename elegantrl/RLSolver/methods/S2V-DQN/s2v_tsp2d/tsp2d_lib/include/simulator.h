#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <vector>
#include <random>
#include "graph.h"

int arg_max(int n, const double* scores);
int arg_min(int n, const double* scores);
double max(int n, const double* scores);

class IEnv;
class Simulator
{
public:
    static void Init(int _num_env);

    static void run_simulator(int num_seq, double eps);

    static int make_action(int num_nodes, std::vector<double>& scores);

    static std::vector<IEnv*> env_list;
    static std::vector< std::shared_ptr<Graph> > g_list;
    static std::vector< std::vector<int>* > covered;
    static std::vector< std::vector<double>* > pred;

    static std::default_random_engine generator;
    static std::uniform_real_distribution<double> distribution;
};

#endif