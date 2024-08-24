#include "nstep_replay_mem.h"
#include "i_env.h"
#include "config.h"
#include <cassert>

#define max(x, y) (x > y ? x : y)

std::vector< std::shared_ptr<Graph> > NStepReplayMem::graphs;
std::vector<int> NStepReplayMem::actions;
std::vector<double> NStepReplayMem::rewards;
std::vector< std::vector<int> > NStepReplayMem::states;
std::vector< std::vector<int> > NStepReplayMem::s_primes;
std::vector<bool> NStepReplayMem::terminals;
int NStepReplayMem::current;
int NStepReplayMem::count;
int NStepReplayMem::memory_size;
std::default_random_engine NStepReplayMem::generator;
std::uniform_int_distribution<int>* NStepReplayMem::distribution;

void NStepReplayMem::Init(int _memory_size)
{
    memory_size = _memory_size;
    graphs.resize(memory_size);
    actions.resize(memory_size);
    rewards.resize(memory_size);
    states.resize(memory_size);
    s_primes.resize(memory_size);
    terminals.resize(memory_size);

    current = 0;
    count = 0;
    distribution = new std::uniform_int_distribution<int>(0, memory_size - 1);
}

void NStepReplayMem::Clear()
{
    current = count = 0;
}

void NStepReplayMem::Add(std::shared_ptr<Graph> g, 
                        std::vector<int>& s_t,
                        int a_t, 
                        double r_t,
                        std::vector<int>& s_prime,
                        bool terminal)
{
    graphs[current] = g;
    actions[current] = a_t;
    rewards[current] = r_t;
    states[current] = s_t;
    s_primes[current] = s_prime;
    terminals[current] = terminal;

    count = max(count, current + 1);
    current = (current + 1) % memory_size; 
}

void NStepReplayMem::Add(IEnv* env)
{
    assert(env->isTerminal());
    int num_steps = env->state_seq.size();
    assert(num_steps);

    env->sum_rewards[num_steps - 1] = env->reward_seq[num_steps - 1];
    for (int i = num_steps - 1; i >= 0; --i)
        if (i < num_steps - 1)
            env->sum_rewards[i] = env->sum_rewards[i + 1] + env->reward_seq[i];

    for (int i = 0; i < num_steps; ++i)
    {
        bool term_t = false;
        double cur_r;
        std::vector<int>* s_prime; 
        if (i + cfg::n_step >= num_steps)
        {
            cur_r = env->sum_rewards[i];
            s_prime = &(env->action_list);
            term_t = true;
        } else {
            cur_r = env->sum_rewards[i] - env->sum_rewards[i + cfg::n_step];
            s_prime = &(env->state_seq[i + cfg::n_step]);
        }
        Add(env->graph, env->state_seq[i], env->act_seq[i], cur_r, *s_prime, term_t);
    }
}

void NStepReplayMem::Sampling(int batch_size, ReplaySample& result)
{
    assert(count >= batch_size);

    result.g_list.resize(batch_size);
    result.list_st.resize(batch_size);
    result.list_at.resize(batch_size);
    result.list_rt.resize(batch_size);
    result.list_s_primes.resize(batch_size);
    result.list_term.resize(batch_size);
    auto& dist = *distribution;
    for (int i = 0; i < batch_size; ++i)
    {
        int idx = dist(generator) % count;
        result.g_list[i] = graphs[idx];
        result.list_st[i] = &(states[idx]);
        result.list_at[i] = actions[idx];
        result.list_rt[i] = rewards[idx];
        result.list_s_primes[i] = &(s_primes[idx]);
        result.list_term[i] = terminals[idx];
    }
}
