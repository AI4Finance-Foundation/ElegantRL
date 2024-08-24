#ifndef NSTEP_REPLAY_MEM_H
#define NSTEP_REPLAY_MEM_H

#include <vector>
#include <random>
#include "graph.h"

class IEnv;

class ReplaySample
{
public:

    std::vector< std::shared_ptr<Graph> > g_list;
    std::vector< std::vector<int>* > list_st, list_s_primes;
    std::vector<int> list_at;
    std::vector<double> list_rt;
    std::vector<bool> list_term;
};

class NStepReplayMem
{
public:
    static void Init(int memory_size);

    static void Add(std::shared_ptr<Graph> g, 
                    std::vector<int>& s_t,
                    int a_t, 
                    double r_t,
                    std::vector<int>& s_prime,
                    bool terminal);

    static void Add(IEnv* env);

    static void Sampling(int batch_size, ReplaySample& result);

    static void Clear();

    static std::vector< std::shared_ptr<Graph> > graphs;
    static std::vector<int> actions;
    static std::vector<double> rewards;
    static std::vector< std::vector<int> > states, s_primes;
    static std::vector<bool> terminals;

    static int current, count, memory_size;
    static std::default_random_engine generator;
    static std::uniform_int_distribution<int>* distribution;
};

#endif