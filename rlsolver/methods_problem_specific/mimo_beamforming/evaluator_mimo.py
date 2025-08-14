import torch
import pickle as pkl
from rlsolver.problems.mimo_beamforming.net_mimo import Policy_Net_MIMO
from rlsolver.problems.mimo_beamforming.mimo_beamforming_env.env_mimo import MIMOEnv

def evaluator(policy_net_mimo, K=4, N=4, M=4, P=10, noise_power=1, evaluate_H_path="./Channel_K=4_N=4_P=10_Samples=120_Optimal=9.9.pkl", device=torch.device("cpu")):
    env_mimo = MIMOEnv(K=K, N=N, M=M, P=P, noise_power=noise_power, device=device, num_env=1000)
    with open(evaluate_H_path, 'rb') as f:
        evaluate_H = torch.as_tensor(pkl.load(f), dtype=torch.cfloat).to(device) 
    state = env_mimo.reset(if_test=True, test_H=evaluate_H)
    sum_rate = torch.zeros(state[0].shape[0], env_mimo.episode_length, 1)
    while(1):
        action = policy_net_mimo(state)
        next_state, reward, done = env_mimo.step(action)
        sum_rate[:, env_mimo.num_steps-1] = reward
        state = next_state
        if done:
            break
    print(f"test_sum_rate: {sum_rate.max(dim=1)[0].mean().item()}")

if __name__  == "__main__":
    N = 4   #   #antennas
    K = 4   #   #users
    P = 10  # power constraint
    noise_power = 1
    
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy_net_mimo = Policy_Net_MIMO().to(device)
    trained_model_path = "rl_cl_sum_rate_9.77_trained_network.pth"
    policy_net_mimo.load_state_dict(torch.load(trained_model_path, map_location=device))
    evaluate_path = "Channel_K=4_N=4_P=10_Samples=120_Optimal=9.9.pkl"
    
    evaluator(policy_net_mimo, K=K, N=N, device=device, P=P, noise_power=noise_power, evaluate_path=evaluate_path)
