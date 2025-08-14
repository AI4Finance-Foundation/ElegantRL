import os
import torch as th
import sys
from net_mimo import Policy_Net_MIMO
from env_mimo import MIMOEnv
import pickle as pkl
from tqdm import tqdm 
import wandb
import time
reward_mode = ['empirical', 'analytical', 'supervised_mmse', 'rl', 'supervised_mmse_curriculum']
def train_curriculum_learning(policy_net_mimo, optimizer, save_path, device, K=4, N=4, P=10, noise_power=1, num_epochs=1000000,
                    num_epochs_per_subspace=1200, num_epochs_to_save_model=1000, num_env=512, epoch_end_switch=20000):
    env_mimo_relay = MIMOEnv(K=K, N=N, P=P, noise_power=noise_power, device=device, num_env=num_env, reward_mode=reward_mode[int(sys.argv[1])], episode_length=6)
    pbar = tqdm(range(num_epochs))
    sum_rate = th.zeros(100, env_mimo_relay.episode_length, 2)
    sum_rate_train = th.zeros(num_env, env_mimo_relay.episode_length, 1)
    test_P = [10 ** 1, 10 ** 2]
    start_time = time.time()
    for epoch in pbar:
        state = env_mimo_relay.reset()
        loss = 0
        sr = 0
        while(1):
            action = policy_net_mimo(state)
            next_state, reward, done, _ = env_mimo_relay.step(action)
            if env_mimo_relay.reward_mode == "rl":
                loss -= reward.mean()
            else:
                loss -= reward.mean()
            if (epoch + 1) % 5 == 0:
                sum_rate_train[:,  env_mimo_relay.num_steps-1, 0] = _.squeeze()
            state = next_state
            if done:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                break
        if epoch % 20 == 0 and env_mimo_relay.reward_mode == 'supervised_mmse_curriculum':
            env_mimo_relay.epsilon = min(1, (epoch / epoch_end_switch))
        if os.path.isfile(os.path.join(save_path, "change_to_sr")):
            env_mimo_relay.reward_mode = "rl"
        if epoch == epoch_end_switch:
            env_mimo_relay.reward_mode = "rl"
        if (epoch+1) % num_epochs_to_save_model == 0 and if_save:
            th.save(policy_net_mimo.state_dict(), save_path + f"{epoch}.pth")
        if (epoch + 1) % num_epochs_per_subspace == 0 and env_mimo_relay.subspace_dim <= 2 * K * N:
            env_mimo_relay.subspace_dim +=  int((2 * K * N)/ 128)
        if (epoch+1) % 5 == 0:
            with th.no_grad():
                for i_p in range(2):
                    state = env_mimo_relay.reset(test=True, test_P = test_P[i_p])
                    while(1):
                        action = policy_net_mimo(state)
                        next_state, _, done, reward = env_mimo_relay.step(action)
                        sum_rate[:, env_mimo_relay.num_steps-1, i_p] = reward.squeeze()
                        state = next_state
                        if done:
                            break
                description = f"id: {epoch} | test_sum_rate_SNR=10: {sum_rate[:, :, 0].max(dim=1)[0].mean()} | test_sum_rate_SNR=20:{sum_rate[:, :, 1].max(dim=1)[0].mean()}| training_loss: {loss.mean(). item() / env_mimo_relay.episode_length:.3f} | gpu memory:{th.cuda.memory_allocated():3d} | elapsed_time:{time.time()-start_time}"
                pbar.set_description(description)
                if if_save:
                    log_path = open(os.path.join(save_path, 'logs','log.txt'), "a+")
                    log_path.write(description + '\n')
                    log_path.close()
            if if_wandb:
                wandb.log({f"train_sum_rate_SNR=10": sum_rate_train[:, :, 0].max(dim=1)[0].mean(), f"test_sum_rate_SNR=10": sum_rate[:, :, 0].max(dim=1)[0].mean(),f"test_sum_rate_SNR=20": sum_rate[:, :, 1].max(dim=1)[0].mean(),"training_loss": loss.mean().item() / env_mimo_relay.episode_length, "elapsed_time": (time.time()-start_time)})
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
    os.mkdir('./{}/{}/{}/'.format(env_name, max_exp_id, 'source_code'))
    os.mkdir('./{}/{}/{}/'.format(env_name, max_exp_id, 'logs'))
    return f"./{env_name}/{max_exp_id}/"

if __name__  == "__main__":

    N = K = int(sys.argv[2])
    SNR = 10
    P = 10 ** (SNR / 10)
    mid_dim = 1024
    noise_power = 1
    learning_rate = 5e-5
    cwd = f"{reward_mode[int(sys.argv[1])]}_H_CL_REINFORCE_N{N}K{K}SNR{SNR}"
    env_name = f"RANDOM_N{N}K{K}SNR{SNR}_mimo_beamforming"
    save_path = None
    if_save = True
    if if_save == True:
        save_path = get_cwd(env_name) # cwd (current work directory): folder to save the trained policy net
        import shutil
        file_list = ["env_mimo.py", "net_mimo.py", "train.py"]
        for file in file_list:
            shutil.copy2(file, os.path.join(save_path,'source_code'))
    device=th.device("cuda:0" if th.cuda.is_available() else "cpu")
    policy_net_mimo = Policy_Net_MIMO(mid_dim= mid_dim, K=K, N=N, P=P).to(device)
    optimizer = th.optim.Adam(policy_net_mimo.parameters(), lr=learning_rate)

    config = {
        'method': 'REINFORCE',
        'objective': reward_mode[int(sys.argv[1])],
        'SNR': SNR,
        'mid_dim': mid_dim,
        'num_subspace_dim_update': 2,
        'path': save_path,
        'num_env': 1024
    }
    if_wandb = True
    if if_wandb:
        wandb.init(
            project=f'REINFORCE_' + 'H' + f'_N{N}K{K}',
            entity="beamforming",
            sync_tensorboard=True,
            config=config,
            name=cwd,
            monitor_gym=True,
            save_code=True,
        )
    try:
        train_curriculum_learning(policy_net_mimo, optimizer, K=K, N=N, save_path=save_path, device=device, P=P, noise_power=noise_power)
        if if_save:
            th.save(policy_net_mimo.state_dict(), save_path + "policy_net_mimo_1.pth")  # number your result policy net
            print(f"saved at " + save_path)
    except KeyboardInterrupt:
        if if_save:
            th.save(policy_net_mimo.state_dict(), save_path + "policy_net_mimo_1.pth")  # number your result policy net
            print(f"saved at " + save_path)
        exit()

