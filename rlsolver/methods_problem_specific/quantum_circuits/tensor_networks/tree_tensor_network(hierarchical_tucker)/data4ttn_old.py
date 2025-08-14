import torch as th
import torch
device = th.device("cuda:0")
# Number of tensors, to (the nth power of 2) - 1
N = 31

def power_of_two(n):
    count = 0
    while n > 1:
        if n % 2 == 1:
            return None
        count += 1
        n //= 2
    return count

X = (power_of_two(N + 1))
num_env = 100
max_dim = 2
test_state = torch.ones((num_env, N + 2, N + 2), device=device).to(torch.float32)
# test_state = torch.randint(0,max_dim, (num_env, N + 2, N + 2), device=device).to(torch.float32)
mask = th.zeros(N + 2, N + 2).to(device)
for i in range(1, 2 ** (X - 1)):
    mask[i*2, i] = 1
    mask[i*2+1, i] = 1
mask = mask.reshape(-1).repeat(1, num_env).reshape(num_env, N + 2, N + 2).to(device)
test_state = th.mul(test_state, mask)
test_state += th.ones_like(test_state)
# print(test_state)
with open(f"test_data_tensor_ring_N={N}.pkl", 'wb') as f:
    import pickle as pkl
    pkl.dump(test_state, f)
