import numpy as np
import matplotlib.pyplot as plt

def discounted_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_sum = 0
    for i in reversed(range(0,len(r))):
        discounted_r[i] = running_sum * gamma + r[i]
        running_sum = discounted_r[i]
    return list(discounted_r)


def moving_avg(a, window_size):
    new_a = []
    for i in range(len(a)):
        if i < window_size:
            new_a.append(0)
        else:
            new_a.append(np.mean(a[i-window_size:i]))
    return new_a


def plot_arr(arr, label = None, window_size = 101):
    arr = np.array(arr)
    if window_size > 1:
        arr = moving_avg(arr, window_size)
    plt.plot(np.arange(len(arr)), arr, label = label)
    plt.title(f"Moving Average Reward, Window Size {window_size}")

def get_reward_sums(filepath):
    try:
        f = open(filepath, 'r')
    except:
        print("Could not find a historical run of the env_config and policy_params.")
        raise FileNotFoundError

    reward_sums = []
    for i, reward_sum in enumerate(f):
        if i < 4: # skip parameter descriptions
            continue
        try:
            reward_sums.append(float(reward_sum))
        except ValueError:
            continue
    return reward_sums






