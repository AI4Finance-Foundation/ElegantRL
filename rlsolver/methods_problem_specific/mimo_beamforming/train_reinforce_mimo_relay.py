import os
import torch
from rlsolver.problems.mimo_beamforming.mimo_beamforming_env.env_mimo_relay import MIMORelayEnv
from rlsolver.problems.mimo_beamforming.net_mimo_relay import Policy_Net_MIMO_Relay
from rlsolver.problems.mimo_beamforming.evaluator_mimo_relay import evaluator_relay

def train_curriculum_learning_relay(policy_net_mimo_relay, optimizer, device, save_path=None, K=4, N=4, M=4, P=10, noise_power=1, num_epochs=400000,
                num_epochs_per_subspace=1000, num_epochs_to_save_model=1000, num_epochs_to_evaluate=100):
    env_mimo = MIMORelayEnv(K=K, N=N, M=M, P=P, noise_power=noise_power, device=device, num_env=4096)
    for epoch in range(num_epochs):
        state = env_mimo.reset()
        loss = 0
        while(1):
            action = policy_net_mimo_relay(state)
            next_state, reward, done = env_mimo.step(action)
            loss -= reward.mean()
            state = next_state
            if done:
                break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f" training_loss: {loss.mean().item():.3f} | gpu memory: {torch.cuda.memory_allocated():3d}")
        if (epoch + 1) % num_epochs_to_save_model == 0:
            torch.save(policy_net_mimo_relay.state_dict(), save_path + f"{epoch}.pth")    
        if (epoch + 1) % num_epochs_evaluate == 0:
            evaluator_relay(policy_net_mimo_relay, device, K=K, N=N, M=M, P=P)
            
def get_cwd(env_name):
    file_list = os.listdir()
    if env_name not in file_list:
        os.mkdir(env_name)
    file_list = os.listdir('./{}/'.format(env_name))
    max_exp_id = 0
    for exp_id in file_list:
        if int(exp_id) + 1 > max_exp_id:
            max_exp_id = int(exp_id) + 1
    os.mkdir('./{}/{}/'.format(env_name, max_exp_id))
    return f"./{env_name}/{max_exp_id}/"

if __name__  == "__main__":
    N = 4   # number of antennas
    K = 4   # number of users
    M = 4 
    P = 10  # power constraint
    noise_power = 1
    learning_rate = 5e-5
    
    env_name = "mimo_beamforming_relay"
    save_path = get_cwd(env_name) # cwd (current work directory): folder to save the trained policy net
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy_net_mimo_relay = Policy_Net_MIMO_Relay(K=K, N=N, M=M).to(device)
    optimizer = torch.optim.Adam(policy_net_mimo_relay.parameters(), lr=learning_rate)
    
    try:
        train_curriculum_learning_relay(policy_net_mimo_relay, optimizer, K=K, N=N, M=M,  device=device, P=P, noise_power=noise_power)
        torch.save(policy_net_mimo_relay.state_dict(), save_path + "policy_net_mimo_1.pth")  # number your result policy net
    except KeyboardInterrupt:
        torch.save(policy_net_mimo_relay.state_dict(), save_path + "policy_net_mimo_1.pth")  # number your result policy net
        exit()

