import torch
import numpy as np
import copy
from copy import deepcopy
import time
import wandb

# Set variables 
nr_of_users = 4
nr_of_BS_antennas = 4

epsilon = 0.0001 # used to end the iterations of the WMMSE algorithm in Shi et al. when the number of iterations is not fixed (note that the stopping criterion has precendence over the fixed number of iterations)
power_tolerance = 0.0001 # used to end the bisection search in the WMMSE algorithm in Shi et al.
total_power = 10 # power constraint in the weighted sum rate maximization problem 
noise_power = 1
path_loss_option = False # used to add a random path loss (drawn from a uniform distribution) to the channel of each user
path_loss_range = [-5,5] # interval of the uniform distribution from which the path loss if drawn (in dB)
nr_of_batches_training = 40000 # used for training
nr_of_batches_test = 1#1000 # used for testing
nr_of_samples_per_batch = 120
nr_of_iterations = 1 # for WMMSE algorithm in Shi et al. 
nr_of_iterations_nn = 1 # for the deep unfolded WMMSE in our paper

# User weights in the weighted sum rate (denoted by alpha in our paper)
user_weights = np.reshape(np.ones(nr_of_users*nr_of_samples_per_batch),(nr_of_samples_per_batch,nr_of_users,1))
user_weights = torch.as_tensor(user_weights)
user_weights_for_regular_WMMSE = np.ones(nr_of_users)
#Q = torch.randn(32,32,dtype=torch.float)
Q = torch.rand(32,32,dtype=torch.float)
q, _ = torch.linalg.qr(Q)
#M = torch.randn(32, 32, dtype=torch.float)
M = torch.rand(32, 32, dtype=torch.float)
q_, _ = torch.linalg.qr(M)
print(q.sum(dim=1))

subspace = 1
total_steps = 0


learning_rate = 5e-5
batch_size = 8192
#gpu_id = 0
gamma = 0.99
mid_dim = 512
unfold_loop = 5
#cwd = "USL_N4K4P10"
cwd = "Dropout_Podracer_H_CL_REINFORCE_N4K4P10"
config = {
    'method': 'REINFORCE',
    'learning_rate': learning_rate,
    'batch_size': batch_size,
    'mid_dim' : mid_dim,
    'gamma': gamma,
    'SNR': total_power,
    'unfold_loop': unfold_loop
}
wandb_ = True
if wandb_:
    wandb.init(
        project='Podracer_' + 'H' + '_N4K4P10',
        entity="beamforming",
        sync_tensorboard=True,
        config=config,
        name=cwd,
        monitor_gym=True,
        save_code=True,
    )
 
def compute_channel(num_antennas, num_users, batch_size , total_power,total_steps, subspace, q, q_, path_loss_option = False, path_loss_range = [-5,5], std=1.0, test=False):
  
  h_ = torch.randn(batch_size, subspace, 1) + 1e-9
  #h_ = h_ / (2)**0.5
  H_CL_ = torch.bmm(q[:subspace].T.repeat(batch_size, 1).reshape(batch_size, q.shape[1], subspace), h_)#.reshape(-1, 2,4,4)
  H_CL_ = torch.bmm(q_.T.repeat(batch_size, 1).reshape(batch_size, 32, 32), H_CL_).reshape(-1 ,2, 4, 4)
  H_CL = H_CL_[:, 0] + H_CL_[:, 1] * 1.j
  #print(torch.isnan(H_CL))
  H_CL = (H_CL * ( 32 / subspace) ** 0.5).reshape(-1, 4 * 4)
  #print(H_CL.shape)
  #print(torch.isnan(H_CL))
  H_CL = ((num_antennas * num_users) ** 0.5) * (H_CL / H_CL.norm(dim=1, keepdim=True))
  #print(torch.isnan(H_CL))
  #print(H_CL.shape) 
  #assert 0
  return H_CL.reshape(-1, 4, 4)

def save(net):
  import os
  file_list = os.listdir()
  folder_name = f"lr_{learning_rate}_bs_{batch_size}_middim_{mid_dim}_gamma_{gamma}"
  if folder_name not in file_list:
    os.mkdir(folder_name)
  file_list = os.listdir('./{}/'.format(folder_name))
  
  exp_id = 0

  for name in file_list:
    exp_id_ = int(name)
    if exp_id_+1 > exp_id:
      exp_id = exp_id_ + 1
  print("Finished experiment {}, {}.".format(folder_name, exp_id))


  os.mkdir('./{}/{}/'.format(folder_name, exp_id))
  path = './{}/{}/net.pth'.format(folder_name, exp_id)
  torch.save(net.state_dict(), path)

if __name__  == "__main__":
  WSR_WMMSE =[] # to store the WSR attained by the WMMSE
  WSR_ZF = [] # to store the WSR attained by the zero-forcing 
  WSR_RZF = [] # to store the WSR attained by the regularized zero-forcing
  WSR_nn = [] # to store the WSR attained by the deep unfolded WMMSE
  WSR_max = []
  WSR_mean = []
  WSR_last = []
  training_loss = []
  agent_num = 10
  from net825 import MMSE_Net
  from net825 import weights_init_uniform
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
  #net_target = net.MMSE_Net(mid_dim)
  #net_target.to(device)
  #net_target.load_state_dict(torch.load('./net.pth', map_location=torch.device("cuda:0")))
  mmse_net_list = [MMSE_Net(mid_dim).to(device) for _ in range(agent_num)]#.to( torch.device('cuda:0'))
  optimizer_list = [torch.optim.Adam(mmse_net_list[i].parameters(), lr=learning_rate) for i in range(agent_num)]
  #scheduler_list = [torch.optim.lr_scheduler.StepLR(optimizer_list[i], step_size=100, gamma=gamma) for i in range(agent_num)]
  print("start of session")
  start_of_time = time.time()
  all_step = []
  import pickle as pkl
  
  with open("./Channel_K=4_N=4_P=10_Samples=120_Optimal=9.9.pkl", 'rb') as f:
    H = torch.as_tensor(pkl.load(f)).to(device)
  try:
    from tqdm import tqdm
    #for i in tqdm(range(1)):
    WSR_last = torch.zeros(10).to(device)
    WSR_mean = torch.zeros(10).to(device)
    WSR_max = torch.zeros(10).to(device)
    WSR_loss = torch.zeros(10).to(device)
    pbar = tqdm(range(nr_of_batches_training))
    #print("traing_loss  wsr: ", WSR_loss.mean().item() / batch_size, " 120 samples: ",WSR_last.max().item(), WSR_max.max().item(), WSR_mean.max().item())
    for i in pbar:
      pbar.set_description(f" training_loss: { WSR_loss.mean().item() / batch_size:.3f} | last: {WSR_last.max().item():.3f} | max: {WSR_max.max().item():.3f} | mean: { WSR_mean.max().item():.3f} | gpu memory: {torch.cuda.memory_allocated():3d}")
      batch_for_training = []
      initial_transmitter_precoder_batch = []
      mmse_net_input = []
      mmse_net_target = []
      if (total_steps+1) % 400 == 0:
        subspace +=1
        
        if subspace > 32:
          subspace = 32
        #print(WSR_max.detach().cpu().numpy() / WSR_max.sum().item())
        #indices = np.random.choice([i for i in range(agent_num)], size=2, replace=False, p = WSR_mean.detach().cpu().numpy() / WSR_mean.sum().item())
        #print(indices)
        #assert 0
        value, indices = WSR_mean.sort(descending=True)
        
        mmse_net_1 = deepcopy(mmse_net_list[indices[0]])
        mmse_net_2 = deepcopy(mmse_net_list[indices[1]])  
        optim_1 = deepcopy(optimizer_list[indices[0]])
        optim_2 = deepcopy(optimizer_list[indices[1]])
        for agent_id in range(5):
            mmse_net_list[agent_id].load_state_dict(mmse_net_1.state_dict())
            mmse_net_list[agent_id + 5].load_state_dict(mmse_net_2.state_dict())
            optimizer_list[agent_id].load_state_dict(optim_1.state_dict())
            optimizer_list[agent_id + 5].load_state_dict(optim_2.state_dict())
        del mmse_net_1
        del mmse_net_2
        del optim_1
        del optim_2
        WSR_last = torch.zeros(10).to(device)
        WSR_mean = torch.zeros(10).to(device)
        WSR_max = torch.zeros(10).to(device)
        WSR_loss = torch.zeros(10).to(device)


      total_steps += 1
      for agent_id in range(agent_num):
        if subspace == 32:
          mmse_net_input = (torch.randn(batch_size, nr_of_BS_antennas, nr_of_users, dtype=torch.cfloat)).to(device)
        else:
          mmse_net_input = compute_channel(nr_of_BS_antennas, nr_of_users, batch_size, total_power,total_steps, subspace, q, q_, path_loss_option, path_loss_range, std = 1.0, test=False)
        W = mmse_net_input
        
        mmse_net_input = torch.as_tensor(mmse_net_input).to(device)
        
        mmse_net_target = mmse_net_list[agent_id].calc_mmse(mmse_net_input)
        mmse_net_target = torch.as_tensor(mmse_net_target).to(device)
        #mmse_net_target = torch.randn(batch_size, 4, 4, dtype=torch.cfloat, device = device)
        tmp = mmse_net_input.to(torch.cfloat)
        mmse_net_target = mmse_net_target.to(torch.cfloat)
        mmse_net_target_ = mmse_net_target 
        mmse_net_input= torch.cat((torch.as_tensor(mmse_net_input.real).reshape(-1, 16), torch.as_tensor(mmse_net_input.imag).reshape(-1, 16)), 1)
        mmse_net_input = torch.as_tensor(mmse_net_input, dtype=torch.float32).to(device)
        initial_tp = torch.as_tensor(np.array(initial_transmitter_precoder_batch), dtype=torch.float32)
        t_0 = mmse_net_target[0].cpu().numpy()
        h_w_input = torch.bmm(tmp, mmse_net_target.transpose(1,2).conj())
        h = h_w_input
        mmse_net_target= torch.cat((torch.as_tensor(mmse_net_target.real).reshape(-1, 16), torch.as_tensor(mmse_net_target.imag).reshape(-1, 16)), 1)
        mmse_net_target = torch.as_tensor(mmse_net_target, dtype=torch.float32).to(device)
        h_w_input= torch.cat((torch.as_tensor(h_w_input.real).reshape(-1, 16), torch.as_tensor(h_w_input.imag).reshape(-1, 16)), 1)
        h_w_input = torch.as_tensor(h_w_input, dtype=torch.float32).to(device)
        mmse_net_output = mmse_net_list[agent_id](torch.cat((mmse_net_input, mmse_net_target, h_w_input), 1), h, tmp)
        
        obj = 0
        obj_mse = 0
        MSE_LOSS = torch.nn.MSELoss()
        if i < 0:
            obj_mse += MSE_LOSS(mmse_net_output, mmse_net_target) * 1000
        precoder = mmse_net_output.reshape(batch_size, 2, 16)
        precoder = precoder[:, 0] + precoder[:, 1] * 1j
        precoder = precoder.reshape(-1, 4,4)
        #precoder = mmse_net_target_ + precoder
        precoder = (precoder.reshape(-1, 16) / (1e-9 + precoder.reshape(-1, 16).norm(dim=1, keepdim=True))).reshape(-1, 4, 4) * np.sqrt(10)
        #print(W)
        #assert 0
        obj -= mmse_net_list[agent_id].calc_wsr(user_weights_for_regular_WMMSE, torch.as_tensor(W).to(device), precoder, noise_power, scheduled_users[0])
        #weight = 0.00001 / 15
        #for iu in range(len(scheduled_users)): 
        #  obj -= weight * mmse_net.calc_wsr(user_weights_for_regular_WMMSE, torch.as_tensor(np.array(W)).to(device), precoder, noise_power, scheduled_users[iu])
          #weight = 0.0001 / 15
        weight = 1
        for _ in range(unfold_loop):
          mmse_net_output = mmse_net_list[agent_id](torch.cat((mmse_net_input, mmse_net_output.detach(), h_w_input.detach()), 1), h, tmp)
          precoder_ = mmse_net_output.reshape(batch_size, 2, 16)
          precoder_ = precoder[:, 0] + precoder[:, 1] * 1j
          precoder_ = precoder.reshape(-1, 4,4)
          precoder = precoder_ #+ precoder
          h_w_input = torch.bmm(tmp, precoder.transpose(1,2).conj())
          h = h_w_input
          h_w_input= torch.cat((torch.as_tensor(h_w_input.real).reshape(-1, 16), torch.as_tensor(h_w_input.imag).reshape(-1, 16)), 1)
          h_w_input = torch.as_tensor(h_w_input, dtype=torch.float32).to(device)
          obj -= weight * mmse_net_list[agent_id].calc_wsr(user_weights_for_regular_WMMSE, torch.as_tensor(W).to(device), precoder, noise_power, scheduled_users[0])
        optimizer_list[agent_id].zero_grad()
        WSR_loss[agent_id] = obj.sum().detach().cpu().item() / 6
        obj_weight = 1
        (obj.sum() * obj_weight).backward()
        optimizer_list[agent_id].step()
        print(f"epoch: {i} id: {agent_id} loss: {obj.sum().detach().cpu().item() / batch_size}")
        if i % 5 == 0:
          # Building a batch for testing
          with torch.no_grad(): 
            batch_for_testing = [] 
            mmse_net_target = mmse_net_list[agent_id].calc_mmse(H)
            mmse_net_input = torch.as_tensor(H).to(device)
            mmse_net_target = torch.as_tensor(mmse_net_target).to(device)
            tmp = mmse_net_input.to(torch.cfloat)
            
            mmse_net_target = mmse_net_target.to(torch.cfloat)
            mmse_net_input= torch.cat((torch.as_tensor(mmse_net_input.real).reshape(-1, 16), torch.as_tensor(mmse_net_input.imag).reshape(-1, 16)), 1)
            mmse_net_input = torch.as_tensor(mmse_net_input, dtype=torch.float32).to(device)
            h_w_input = torch.bmm(tmp, mmse_net_target.transpose(1,2).conj())
            
            h = h_w_input
            h_w_input= torch.cat((torch.as_tensor(h_w_input.real).reshape(-1, 16), torch.as_tensor(h_w_input.imag).reshape(-1, 16)), 1)
            h_w_input = torch.as_tensor(h_w_input, dtype=torch.float32).to(device)
            mmse_net_target_ = mmse_net_target
            mmse_net_target= torch.cat((torch.as_tensor(mmse_net_target.real).reshape(-1, 16), torch.as_tensor(mmse_net_target.imag).reshape(-1, 16)), 1)
            mmse_net_target = torch.as_tensor(mmse_net_target, dtype=torch.float32).to(device)
            output = mmse_net_list[agent_id](torch.cat((mmse_net_input, mmse_net_target, h_w_input), 1), h, tmp)
            precoder = output.detach().reshape(-1, 2, 16)
            precoder = precoder[:, 0] + precoder[:, 1] * 1j
            precoder = precoder.reshape(-1, 4,4)
            wsr = torch.zeros(H.shape[0], unfold_loop + 1, 1)
            precoder = (precoder.reshape(-1, 16) / precoder.reshape(-1, 16).norm(dim=1, keepdim=True)).reshape(-1,  4, 4) * np.sqrt(10)
            wsr[:, 0] = mmse_net_list[agent_id].calc_wsr(user_weights_for_regular_WMMSE, H.to(device), precoder, noise_power, scheduled_users[0])
            for _ in range(unfold_loop):
              output = mmse_net_list[agent_id](torch.cat((mmse_net_input, output.detach(), h_w_input.detach()), 1), h, tmp)
              precoder_ = output.reshape(-1, 2, 16)
              precoder_ = precoder_[:, 0] + precoder_[:, 1] * 1j
              precoder_ = precoder_.reshape(-1, 4,4)
              precoder = precoder_ #+ precoder
              h_w_input = torch.bmm(tmp, precoder.transpose(1,2).conj())
              h = h_w_input
              h_w_input= torch.cat((torch.as_tensor(h_w_input.real).reshape(-1, 16), torch.as_tensor(h_w_input.imag).reshape(-1, 16)), 1)
              h_w_input = torch.as_tensor(h_w_input, dtype=torch.float32).to(device)
              wsr[:, _ + 1] = mmse_net_list[agent_id].calc_wsr(user_weights_for_regular_WMMSE, H.to(device), precoder, noise_power, scheduled_users[0])
            
            #print(agent_id, wsr.shape)
            
            #assert 0
            wsr = wsr.reshape(wsr.shape[0], -1)
            WSR_last[agent_id] = wsr[:, -1].mean()
            WSR_mean[agent_id] = wsr.mean(dim=1).mean()
            WSR_max_tmp, _ = wsr.max(dim=1)
            #print(wsr.max(dim=1))
            #assert 0
            WSR_max[agent_id] = WSR_max_tmp.mean()
            #print(WSR_max[agent_id], WSR_last[agent_id], WSR_mean[agent_id])
            #assert 0

      if i % 5 == 0:
        if wandb_:
          wandb.log({'wsr': WSR_loss.mean().item() / batch_size, '120_samples': WSR_last.max(), '120_max_wsr:': WSR_max.max(), '120_mean_wsr:': WSR_mean.max()})

          
  except KeyboardInterrupt:
    
    save(mmse_net_list[0])
    exit()
  save(mmse_net_list[0])
  print("Training took:", time.time()-start_of_time)






