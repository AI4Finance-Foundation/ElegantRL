import torch
from rlsolver.problems.mimo_beamforming.mimo_beamforming_env.env_mimo_relay import MIMORelayEnv


def evaluator_relay(policy_net_mimo_relay, K=4, N=4, M=4, P=10, noise_power=1, evaluate_H_path="H_K4N4M4.pkl", evaluate_G_path="G_K4N4M4.pkl", device=torch.device("cpu")):
    env_mimo = MIMORelayEnv(K=K, N=N, M=M, P=P, noise_power=noise_power, device=device, num_env=1000)
    import pickle as pkl
    with open(evaluate_H_path, 'rb') as f:
        evaluate_H = torch.as_tensor(pkl.load(f), dtype=torch.cfloat).to(device) 
    with open(evaluate_G_path, 'rb') as f:
        evaluate_G = torch.as_tensor(pkl.load(f), dtype=torch.cfloat).to(device)
    state = env_mimo.reset(if_test=True, test_H=evaluate_H, test_G=evaluate_G)
    sum_rate = torch.zeros(state[0].shape[0], env_mimo.episode_length, 1)
    while(1):
        action = policy_net_mimo_relay(state)
        next_state, reward, done = env_mimo.step(action)
        sum_rate[:, env_mimo.num_steps-1] = reward
        state = next_state
        if done:
            break
    print(f"test_sum_rate: {sum_rate.max(dim=1)[0].mean().item()}")
