# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
# Creates a pool of n distributed agents, with thread-safe batched access to    #
# the policy. Runs training episodes and collects and processes transitions.    #
# Based on code by Scavuzzo et al. and adjusted to work with node selection.    #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #

import threading
import queue

from pyscipopt import scip

import util
from util import init_scip_params
from nodesels.nodesel_agent import NodeselAgent


class AgentPool:
    """
    Class holding the reference to the agents and the policy sampler.
    Puts jobs in the queue through job sponsors.
    """

    def __init__(self, brain, n_agents, time_limit, mode):
        self.jobs_queue = queue.Queue()
        self.requests_queue = queue.Queue()
        self.policy_sampler = PolicySampler("Policy BaseSampler", brain, self.requests_queue)
        self.agents = [Agent(f"Agent {i}", time_limit, self.jobs_queue, self.requests_queue, mode)
                       for i in range(n_agents)]

    def start(self):
        self.policy_sampler.start()
        for agent in self.agents:
            agent.start()

    def close(self):
        # order the episode sampling agents to stop
        for _ in self.agents:
            self.jobs_queue.put(None)
        self.jobs_queue.join()
        # order the policy sampler to stop
        self.requests_queue.put(None)
        self.requests_queue.join()

    def start_job(self, instances, sample_rate, static, greedy, block_policy=True):
        """
        Starts a job. A job is a set of tasks.
        A task consists of an instance and instructions (sample rate, greediness).
        The job sponsor is a queue specific to a job. It is the job sponsor who holds the lists of tasks.
        The role of the job sponsor is to keep track of which tasks have been completed.
        The job queue is loaded with references to the job sponsor.
        """
        job_sponsor = queue.Queue()
        samples = []
        stats = []

        policy_access = threading.Event()
        if not block_policy:
            policy_access.set()

        # For each instance in the batch...
        for instance in instances:
            # Create a task description...
            task = {'instance': instance,
                    'sample_rate': sample_rate,
                    'greedy': greedy,
                    'static': static,
                    'samples': samples,
                    'stats': stats,
                    'policy_access': policy_access
                    }
            # ... and add it to the job sponsor
            job_sponsor.put(task)
            self.jobs_queue.put(job_sponsor)

        res = (samples, stats, job_sponsor)
        if block_policy:
            res = (*res, policy_access)

        return res

    def wait_completion(self):
        # wait for all running episodes to finish
        self.jobs_queue.join()


class PolicySampler(threading.Thread):
    """
    Gather policy sampling requests from the agents, and process them in a batch.
    """

    def __init__(self, name, brain, requests_queue):
        super().__init__(name=name)
        self.brain = brain
        self.requests_queue = requests_queue

    def run(self):
        stop_order_received = False
        while not stop_order_received:
            # The goal is to get all the requests from the
            # requests_queue until it is empty and then break
            requests = []
            request = self.requests_queue.get()
            while True:
                # check for a stopping order
                if request is None:
                    self.requests_queue.task_done()
                    stop_order_received = True
                    break
                requests.append(request)  # add request to the batch
                # keep collecting more requests if available, without waiting
                try:
                    request = self.requests_queue.get(block=False)
                except queue.Empty:
                    break

            if not requests: continue
            receivers = [r['receiver'] for r in requests]
            for r in requests: del r['receiver']

            responses = self.brain.sample_actions(requests)
            for receiver, response in zip(receivers, responses):
                self.requests_queue.task_done()
                receiver.put(response)


class Agent(threading.Thread):
    """
    Agent class. Receives tasks from the job sponsor, runs them and samples transitions if requested.
    """

    def __init__(self, name, time_limit, jobs_queue, requests_queue, metric="lb/obj"):
        super().__init__(name=name)
        self.time_limit = time_limit
        self.jobs_queue = jobs_queue
        self.requests_queue = requests_queue
        self.metric = metric

    def run(self):
        while True:
            job_sponsor = self.jobs_queue.get()

            # check for a stopping order
            if job_sponsor is None:
                self.jobs_queue.task_done()
                break

            # Get task from job sponsor
            task = job_sponsor.get()
            instance = task['instance']
            sample_rate = task['sample_rate']
            training = not task['greedy']

            # Run episode
            # Everything from here... -------------------------------------------------------------------
            m = scip.Model()
            m.hideOutput()
            m.readProblem(instance['path'])

            # 1: CPU user seconds, 2: wall clock time
            m.setIntParam('timing/clocktype', 2)
            m.setRealParam('limits/time', self.time_limit)
            init_scip_params(m, instance['seed'], task['static'])
            m.setRealParam('limits/objectivestop', abs(instance['sol']))

            nodesel_agent = NodeselAgent(instance=instance['path'],
                                         opt_sol=instance['sol'],
                                         seed=instance['seed'],
                                         greedy=task['greedy'],
                                         metric=self.metric,
                                         sample_rate=sample_rate,
                                         requests_queue=self.requests_queue)

            m.includeNodesel(nodesel=nodesel_agent,
                             name='nodesel agent',
                             desc='Node selection agent',
                             stdpriority=999999,
                             memsavepriority=999999)

            task['policy_access'].wait()
            m.optimize()
            # ... to here. -------------------------------------------------------------------

            # avoid too large trees during training for stability
            if (nodesel_agent.iter_count > 50000) and training:
                job_sponsor.task_done()
                self.jobs_queue.task_done()
                continue

            # post-process the collected samples (credit assignment)
            if sample_rate > 0:
                if self.metric in ["nnodes", "lb-obj"]:
                    total_penalty = nodesel_agent.penalty
                    for transition in nodesel_agent.transitions:
                        # negative return equals penalty before action - total penalty
                        transition['returns'] = transition['penalty'] - total_penalty
                else:  # self.metric = "gub+"
                    total_reward = 0
                    for transition in nodesel_agent.transitions[::-1]:
                        total_reward += transition['reward']
                        transition['returns'] = total_reward
                        total_reward *= 0.997
                    # subtree_sizes = nodesel_agent.tree_recorder.calculate_subtree_sizes()
                    # for transition in nodesel_agent.transitions:
                    #     transition['returns'] = -subtree_sizes[transition['node_id']] - 1

            # record episode samples and stats
            task['samples'].extend(nodesel_agent.transitions)
            task['stats'].append({'task': task, 'info': nodesel_agent.info})

            # tell both the agent pool and the original task sponsor that the task is done
            job_sponsor.task_done()
            self.jobs_queue.task_done()
