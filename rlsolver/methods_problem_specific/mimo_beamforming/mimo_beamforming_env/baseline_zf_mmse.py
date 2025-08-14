# Import libraries

import torch
import numpy as np
import copy
from copy import deepcopy
import time
import matplotlib.pyplot as plt


# Set variables 
nr_of_users = 8
nr_of_BS_antennas = 4
scheduled_users = [0,1,2,3] # array of scheduled users. Note that we schedule all the users.
epsilon = 0.0001 # used to end the iterations of the WMMSE algorithm in Shi et al. when the number of iterations is not fixed (note that the stopping criterion has precendence over the fixed number of iterations)
power_tolerance = 0.0001 # used to end the bisection search in the WMMSE algorithm in Shi et al.
total_power = 10 # power constraint in the weighted sum rate maximization problem 
noise_power = 1
path_loss_option = False # used to add a random path loss (drawn from a uniform distribution) to the channel of each user
path_loss_range = [-5,5] # interval of the uniform distribution from which the path loss if drawn (in dB)
nr_of_batches_training = 10000 # used for training
nr_of_batches_test = 1#1000 # used for testing
nr_of_samples_per_batch = 8192
nr_of_iterations = 5 # for WMMSE algorithm in Shi et al. 
nr_of_iterations_nn = 5 # for the deep unfolded WMMSE in our paper

# User weights in the weighted sum rate (denoted by alpha in our paper)
user_weights = np.reshape(np.ones(nr_of_users*nr_of_samples_per_batch),(nr_of_samples_per_batch,nr_of_users,1))
user_weights = torch.as_tensor(user_weights)
user_weights_for_regular_WMMSE = np.ones(nr_of_users)

# Compute power for bisection search in the optimization of the transmitter precoder 
# - eq. (18) in the paper by Shi et al.
def compute_P(Phi_diag_elements, Sigma_diag_elements, mu):
  nr_of_BS_antennas = Phi_diag_elements.size
  mu_array = mu*np.ones(Phi_diag_elements.size)
  result = np.divide(Phi_diag_elements,(Sigma_diag_elements + mu_array)**2)
  result = np.sum(result)
  return result


def compute_norm_of_complex_array(x):
  result = np.sqrt(np.sum((np.absolute(x))**2))
  return result


def compute_sinr_th(channel, precoder, noise_power, user_id, selected_users):
    nr_of_users = np.size(channel,0)
    #print(type(precoder))
    numerator = (torch.absolute(torch.matmul(torch.conj(torch.as_tensor(channel[user_id,:], dtype=torch.complex64)),torch.as_tensor(precoder[user_id,:], dtype=torch.complex64))))**2
    #print(type(precoder))
    inter_user_interference = 0
    for user_index in range(nr_of_users):
      if user_index != user_id and user_index in selected_users:
        inter_user_interference = inter_user_interference + (torch.absolute(torch.matmul(torch.conj(torch.as_tensor(channel[user_id,:], dtype=torch.complex64)),torch.as_tensor(precoder[user_index,:], dtype=torch.complex64))))**2
    denominator = noise_power + inter_user_interference

    result = numerator/denominator
    return result

def compute_sinr(channel, precoder, noise_power, user_id, selected_users):
    nr_of_users = np.size(channel,0)
    numerator = (np.absolute(np.matmul(np.conj(channel[user_id,:]),precoder[user_id,:])))**2

    inter_user_interference = 0
    for user_index in range(nr_of_users):
      if user_index != user_id and user_index in selected_users:
        inter_user_interference = inter_user_interference + (np.absolute(np.matmul(np.conj(channel[user_id,:]),precoder[user_index,:])))**2
    denominator = noise_power + inter_user_interference

    result = numerator/denominator
    return result
def compute_user_weights(nr_of_users, selected_users):
  result = np.ones(nr_of_users)
  for user_index in range(nr_of_users):
    if not (user_index in selected_users):
      result[user_index] = 0
  return result


def compute_weighted_sum_rate_th(user_weights, channel, precoder, noise_power, selected_users):
   result = 0
   nr_of_users = np.size(channel,0)
   #print(type(precoder))
   #assert 0
   for user_index in range(nr_of_users):
     if user_index in selected_users:
       user_sinr = compute_sinr_th(channel, precoder, noise_power, user_index, selected_users)
       result = result + user_weights[user_index]*torch.log(1 + user_sinr)
   return result
def compute_weighted_sum_rate(user_weights, channel, precoder, noise_power, selected_users):
   result = 0
   nr_of_users = np.size(channel,0)
   #print(channel.shape)
   #assert 0
   for user_index in range(nr_of_users):
     if user_index in selected_users:
       user_sinr = compute_sinr(channel, precoder, noise_power, user_index, selected_users)
       result = result + user_weights[user_index]*np.log2(1 + user_sinr)
   return result


def compute_sinr_nn(channel, precoder, noise_power, user_id, nr_of_users):

    numerator = torch.sum((torch.matmul(torch.t(channel[user_id]),precoder[user_id]))**2)
    inter_user_interference = 0
    for user_index in range(nr_of_users):
      if user_index != user_id:
        inter_user_interference = inter_user_interference +  torch.sum((torch.matmul(torch.t(channel[user_id]),precoder[user_index]))**2)
    denominator = noise_power + inter_user_interference

    result = numerator/denominator
    return result


def compute_WSR_nn(user_weights, channel, precoder, noise_power, nr_of_users):

   result = 0
   result_arr = []
   for batch_index in range(nr_of_samples_per_batch):
    t = 0
    if batch_index >= channel.shape[0]:
      break
    for user_index in range(nr_of_users):
        user_sinr = compute_sinr_nn(channel[batch_index], precoder[batch_index], noise_power, user_index,nr_of_users)
        
        t = t + user_weights[batch_index][user_index]*(torch.log(1 + user_sinr)/np.log(2.0))
    result_arr.append(t.item())
    result += t
    #print("id: ", batch_index, "WSR: ", t.item())
    #print(t.item())
   
   return result


# Computes a channel realization and returns it in two formats, one for the WMMSE and one for the deep unfolded WMMSE.
# It also returns the initialization value of the transmitter precoder, which is used as input in the computation graph of the deep unfolded WMMSE.
def compute_channel(nr_of_BS_antennas, nr_of_users, total_power, H = None, path_loss_option = False, path_loss_range = [-5,5]):
  channel_nn = []
  initial_transmitter_precoder = []
  channel_WMMSE = np.zeros((nr_of_users, nr_of_BS_antennas)) + 1j*np.zeros((nr_of_users, nr_of_BS_antennas))

  
  for i in range(nr_of_users):
      regularization_parameter_for_RZF_solution = 0
      path_loss = 0 # path loss is 0 dB by default, otherwise it is drawn randomly from a uniform distribution (N.B. it is different for each user)
      if path_loss_option == True:
        path_loss = np.random.uniform(path_loss_range[0],path_loss_range[-1])
        regularization_parameter_for_RZF_solution = regularization_parameter_for_RZF_solution + 1/((10**(path_loss/10))*total_power) # computed as in "MMSE precoding for multiuser MISO downlink transmission with non-homogeneous user SNR conditions" by D.H. Nguyen and T. Le-Ngoc
      if H.shape[0] != 1:
        result_real = H.real[i,:] #np.sqrt(10**(path_loss/10))*np.sqrt(0.5)*np.random.normal(size = (nr_of_BS_antennas,1))
        result_imag  = H.imag[i, :] #np.sqrt(10**(path_loss/10))*np.sqrt(0.5)*np.random.normal(size = (nr_of_BS_antennas,1))
        #print(result_real.shape, result_real.reshape(-1, 1).shape)
        result_real = result_real.reshape(-1, 1)
        result_imag = result_imag.reshape(-1, 1)
      else:
        result_real = np.sqrt(10**(path_loss/10))*np.sqrt(0.5)*np.random.normal(size = (nr_of_BS_antennas,1))
        result_imag = np.sqrt(10**(path_loss/10))*np.sqrt(0.5)*np.random.normal(size = (nr_of_BS_antennas,1))
      #print(result_real.shape)
      #assert 0
      channel_WMMSE[i,:] = np.reshape(result_real,(1,nr_of_BS_antennas)) + 1j*np.reshape(result_imag, (1,nr_of_BS_antennas))
      result_col_1 = np.vstack((result_real,result_imag))
      result_col_2 = np.vstack((-result_imag,result_real))
      result =  np.hstack((result_col_1, result_col_2))
      initial_transmitter_precoder.append(result_col_1)
      channel_nn.append(result)
  initial_transmitter_precoder_array = np.array(initial_transmitter_precoder)
  initial_transmitter_precoder_array = np.sqrt(total_power)*initial_transmitter_precoder_array/np.linalg.norm(initial_transmitter_precoder_array)
  initial_transmitter_precoder = []
  
  for i in range(nr_of_users):
    initial_transmitter_precoder.append(initial_transmitter_precoder_array[i])
  #print(len(channel_nn), channel_nn[0].shape)
  return channel_nn, initial_transmitter_precoder, channel_WMMSE, regularization_parameter_for_RZF_solution


# Computes the zero-forcing solution as in "MMSE precoding for multiuser MISO downlink transmission with non-homogeneous user SNR conditions" by D.H. Nguyen and T. Le-Ngoc
def zero_forcing(channel_realization, total_power):
  
  ZF_solution = np.matmul((np.transpose(channel_realization)),np.linalg.inv(np.matmul(np.conj(channel_realization),(np.transpose(channel_realization)))))
  ZF_solution = ZF_solution*np.sqrt(total_power)/np.linalg.norm(ZF_solution) # scaled according to the power constraint

  return np.transpose(ZF_solution)


# Computes the regularized zero-forcing solution as in "MMSE precoding for multiuser MISO downlink transmission with non-homogeneous user SNR conditions" by D.H. Nguyen and T. Le-Ngoc
def regularized_zero_forcing(channel_realization, total_power, regularization_parameter = 0, path_loss_option = False):
  
  if path_loss_option == False:
    RZF_solution = np.matmul((np.transpose(channel_realization)),np.linalg.inv(np.matmul(np.conj(channel_realization),(np.transpose(channel_realization))) + nr_of_users/total_power*np.eye(nr_of_users, nr_of_users)))
  else:
    RZF_solution = np.matmul((np.transpose(channel_realization)),np.linalg.inv(np.matmul(np.conj(channel_realization),(np.transpose(channel_realization))) + regularization_parameter*np.eye(nr_of_users, nr_of_users)))

  RZF_solution = RZF_solution*np.sqrt(total_power)/np.linalg.norm(RZF_solution) # scaled according to the power constraint

  return np.transpose(RZF_solution)

def run_WMMSE(epsilon, channel, selected_users, total_power, noise_power, user_weights, max_nr_of_iterations, log = False):
    
  nr_of_users = np.size(channel,0)
  nr_of_BS_antennas = np.size(channel,1)
  WSR=[] # to check if the WSR (our cost function) increases at each iteration of the WMMSE
  break_condition = epsilon + 1 # break condition to stop the WMMSE iterations and exit the while
  receiver_precoder = np.zeros(nr_of_users) + 1j*np.zeros(nr_of_users) # receiver_precoder is "u" in the paper of Shi et al. (it's a an array of complex scalars)
  mse_weights = np.ones(nr_of_users) # mse_weights is "w" in the paper of Shi et al. (it's a an array of real scalars)
  transmitter_precoder = np.zeros((nr_of_users, nr_of_BS_antennas)) + 1j*np.zeros((nr_of_users, nr_of_BS_antennas))# transmitter_precoder is "v" in the paper of Shi et al. (it's a complex matrix)
  
  new_receiver_precoder = np.zeros(nr_of_users) + 1j*np.zeros(nr_of_users) # for the first iteration 
  new_mse_weights = np.zeros(nr_of_users) # for the first iteration
  new_transmitter_precoder = np.zeros((nr_of_users, nr_of_BS_antennas)) + 1j*np.zeros((nr_of_users, nr_of_BS_antennas)) # for the first iteration

  
  # Initialization of transmitter precoder
  for user_index in range(nr_of_users):
    if user_index in selected_users:
      transmitter_precoder[user_index,:] = channel[user_index,:]
  transmitter_precoder = transmitter_precoder/np.linalg.norm(transmitter_precoder)*np.sqrt(total_power)
  
  # Store the WSR obtained with the initialized trasmitter precoder    
  WSR.append(compute_weighted_sum_rate(user_weights, channel, transmitter_precoder, noise_power, selected_users))

  # Compute the initial power of the transmitter precoder
  initial_power = 0
  for user_index in range(nr_of_users):
    if user_index in selected_users:
      initial_power = initial_power + (compute_norm_of_complex_array(transmitter_precoder[user_index,:]))**2 
  if log == True:
    print("Power of the initialized transmitter precoder:", initial_power)

  nr_of_iteration_counter = 0 # to keep track of the number of iteration of the WMMSE

  while break_condition >= epsilon and nr_of_iteration_counter<=max_nr_of_iterations:
    
    nr_of_iteration_counter = nr_of_iteration_counter + 1
    if log == True:
      print("WMMSE ITERATION: ", nr_of_iteration_counter)

    # Optimize receiver precoder - eq. (5) in the paper of Shi et al.
    for user_index_1 in range(nr_of_users):
      if user_index_1 in selected_users:
        user_interference = 0.0
        for user_index_2 in range(nr_of_users):
          if user_index_2 in selected_users:
            user_interference = user_interference + (np.absolute(np.matmul(np.conj(channel[user_index_1,:]),transmitter_precoder[user_index_2,:])))**2

        new_receiver_precoder[user_index_1] = np.matmul(np.conj(channel[user_index_1,:]),transmitter_precoder[user_index_1,:]) / (noise_power + user_interference)

    # Optimize mse_weights - eq. (13) in the paper of Shi et al.
    for user_index_1 in range(nr_of_users):
      if user_index_1 in selected_users:

        user_interference = 0 # it includes the channel of all selected users
        inter_user_interference = 0 # it includes the channel of all selected users apart from the current one
        
        for user_index_2 in range(nr_of_users):
          if user_index_2 in selected_users:
            user_interference = user_interference + (np.absolute(np.matmul(np.conj(channel[user_index_1,:]),transmitter_precoder[user_index_2,:])))**2
        for user_index_2 in range(nr_of_users):
          if user_index_2 != user_index_1 and user_index_2 in selected_users:
            inter_user_interference = inter_user_interference + (np.absolute(np.matmul(np.conj(channel[user_index_1,:]),transmitter_precoder[user_index_2,:])))**2
        
        new_mse_weights[user_index_1] = (noise_power + user_interference)/(noise_power + inter_user_interference)

    A = np.zeros((nr_of_BS_antennas,nr_of_BS_antennas))+1j*np.zeros((nr_of_BS_antennas,nr_of_BS_antennas))
    for user_index in range(nr_of_users):
      if user_index in selected_users:
        # hh should be an hermitian matrix of size (nr_of_BS_antennas X nr_of_BS_antennas)
        hh = np.matmul(np.reshape(channel[user_index,:],(nr_of_BS_antennas,1)),np.conj(np.transpose(np.reshape(channel[user_index,:],(nr_of_BS_antennas,1)))))
        A = A + (new_mse_weights[user_index]*user_weights[user_index]*(np.absolute(new_receiver_precoder[user_index]))**2)*hh

    Sigma_diag_elements_true, U = np.linalg.eigh(A)
    Sigma_diag_elements = copy.deepcopy(np.real(Sigma_diag_elements_true))
    Lambda = np.zeros((nr_of_BS_antennas,nr_of_BS_antennas)) + 1j*np.zeros((nr_of_BS_antennas,nr_of_BS_antennas))
    
    for user_index in range(nr_of_users):
      if user_index in selected_users:     
        hh = np.matmul(np.reshape(channel[user_index,:],(nr_of_BS_antennas,1)),np.conj(np.transpose(np.reshape(channel[user_index,:],(nr_of_BS_antennas,1)))))
        Lambda = Lambda + ((user_weights[user_index])**2)*((new_mse_weights[user_index])**2)*((np.absolute(new_receiver_precoder[user_index]))**2)*hh

    Phi = np.matmul(np.matmul(np.conj(np.transpose(U)),Lambda),U)
    Phi_diag_elements_true = np.diag(Phi)
    Phi_diag_elements = copy.deepcopy(Phi_diag_elements_true)
    Phi_diag_elements = np.real(Phi_diag_elements)

    for i in range(len(Phi_diag_elements)):
      if Phi_diag_elements[i]<np.finfo(float).eps:
        Phi_diag_elements[i] = np.finfo(float).eps
      if (Sigma_diag_elements[i])<np.finfo(float).eps:
        Sigma_diag_elements[i] = 0

    # Check if mu = 0 is a solution (eq.s (15) and (16) of in the paper of Shi et al.)
    power = 0 # the power of transmitter precoder (i.e. sum of the squared norm)
    for user_index in range(nr_of_users):
      if user_index in selected_users:
        if np.linalg.det(A) != 0:
          w = np.matmul(np.linalg.inv(A),np.reshape(channel[user_index,:],(nr_of_BS_antennas,1)))*user_weights[user_index]*new_mse_weights[user_index]*(new_receiver_precoder[user_index])
          power = power + (compute_norm_of_complex_array(w))**2

    # If mu = 0 is a solution, then mu_star = 0
    if np.linalg.det(A) != 0 and power <= total_power:
      mu_star = 0
    # If mu = 0 is not a solution then we search for the "optimal" mu by bisection
    else:
      power_distance = [] # list to store the distance from total_power in the bisection algorithm 
      mu_low = np.sqrt(1/total_power*np.sum(Phi_diag_elements))
      mu_high = 0
      low_point = compute_P(Phi_diag_elements, Sigma_diag_elements, mu_low)
      high_point = compute_P(Phi_diag_elements, Sigma_diag_elements, mu_high)

      obtained_power = total_power + 2*power_tolerance # initialization of the obtained power such that we enter the while 

      # Bisection search
      while np.absolute(total_power - obtained_power) > power_tolerance:
        mu_new = (mu_high + mu_low)/2
        obtained_power = compute_P(Phi_diag_elements, Sigma_diag_elements, mu_new) # eq. (18) in the paper of Shi et al.
        power_distance.append(np.absolute(total_power - obtained_power))
        if obtained_power > total_power:
          mu_high = mu_new
        if obtained_power < total_power:
          mu_low = mu_new
      mu_star = mu_new
      if log == True:
        print("first value:", power_distance[0])
        plt.title("Distance from the target value in bisection (it should decrease)")
        plt.plot(power_distance)
        plt.show()

    for user_index in range(nr_of_users):
      if user_index in selected_users:
        new_transmitter_precoder[user_index,:] = np.matmul(np.linalg.inv(A + mu_star*np.eye(nr_of_BS_antennas)),channel[user_index,:])*user_weights[user_index]*new_mse_weights[user_index]*(new_receiver_precoder[user_index]) 

    # To select only the weights of the selected users to check the break condition
    mse_weights_selected_users = []
    new_mse_weights_selected_users = []
    for user_index in range(nr_of_users): 
      if user_index in selected_users:
        mse_weights_selected_users.append(mse_weights[user_index])
        new_mse_weights_selected_users.append(new_mse_weights[user_index])

    mse_weights = deepcopy(new_mse_weights)
    transmitter_precoder = deepcopy(new_transmitter_precoder)
    receiver_precoder = deepcopy(new_receiver_precoder)
    
    #print(transmitter_precoder)
    #transmitter_precoder = np.random.rand(4,4) + np.random.rand(4,4) * 1j
    
    WSR.append(compute_weighted_sum_rate(user_weights, channel, transmitter_precoder, noise_power, selected_users))
    break_condition = np.absolute(np.log2(np.prod(new_mse_weights_selected_users))-np.log2(np.prod(mse_weights_selected_users)))
    
  if log == True:
    plt.title("Change of the WSR at each iteration of the WMMSE (it should increase)")
    plt.plot(WSR,'bo')
    plt.show()

  return transmitter_precoder, receiver_precoder, mse_weights, WSR[-1]

def calc_wsr(channel_input, initial_tp, step_size):
  user_weights = np.reshape(np.ones(nr_of_users*channel_input.shape[0]),(channel_input.shape[0],nr_of_users,1))
  user_weights = torch.as_tensor(user_weights)
  user_weights_for_regular_WMMSE = np.ones(nr_of_users)

  initial_transmitter_precoder = initial_tp
  # The number of step sizes depends on the selected number of PGD layers, the number of elements for each step size initializer depends on the selected number of deep unfolded iterations
  profit = [] # stores the WSR obtained at each iteration
  for loop in range(0,nr_of_iterations_nn):  
    user_interference2 = []
    
    for batch_index in range(channel_input.shape[0]): 
      if batch_index >= channel_input.shape[0]:
        break
      user_interference_single = []
      for i in range(nr_of_users):
        temp = 0.0
        for j in range(nr_of_users):
          temp = temp + torch.sum((torch.matmul(torch.t(torch.as_tensor(channel_input[batch_index, i,:,:])),initial_transmitter_precoder[batch_index,j,:,:]))**2)
        user_interference_single.append(temp + noise_power)
      user_interference2.append(torch.as_tensor(user_interference_single).unsqueeze(0))
    user_interference2 = torch.cat(user_interference2)

    user_interference_exp2 = torch.tile(torch.unsqueeze(torch.tile(torch.unsqueeze(user_interference2,-1),(1,1,2)),-1),(1,1,1,1))
    receiver_precoder_temp = (torch.matmul(channel_input.permute(0,1,3,2),initial_transmitter_precoder))
    # Optimize the receiver precoder 
    #print(receiver_precoder_temp.shape, user_interference_exp2.shape)

    receiver_precoder = torch.divide(receiver_precoder_temp,user_interference_exp2)

    # Optimize the mmse weights 
    self_interference = torch.sum((torch.matmul(channel_input.permute(0,1,3,2),initial_transmitter_precoder))**2, 2)

    inter_user_interference_total = []

    for batch_index in range(channel_input.shape[0]):
      if batch_index >= channel_input.shape[0]:
        break      
      inter_user_interference_temp = []
      for i in range(nr_of_users):
        temp = 0.0
        for j in range(nr_of_users):
          if j != i:
            temp = temp + torch.sum((torch.matmul(torch.t(channel_input[batch_index, i,:,:]),initial_transmitter_precoder[batch_index,j,:,:]))**2)
        inter_user_interference_temp.append(temp + noise_power) # $sum{|(h_i)*H,v_i}|**2 + noise_power$
      inter_user_interference = torch.as_tensor(inter_user_interference_temp).unsqueeze(0).reshape((1,nr_of_users,1)) # Nx1 $sum{|(h_i)*H,v_i}|**2 + noise_power$
      inter_user_interference_total.append(inter_user_interference)
    inter_user_interference_total = torch.cat(inter_user_interference_total)
        
    mse_weights = (torch.divide(self_interference,inter_user_interference_total)) + 1.0
    for step_i in range(step_size.shape[0]):
      channel = channel_input
      # First iteration
      a1_exp = torch.tile(torch.unsqueeze(mse_weights[:,0,:],-1),(1,2*nr_of_BS_antennas,2*nr_of_BS_antennas))
      a2_exp = torch.tile(torch.unsqueeze(user_weights[:,0,:],-1),(1,2*nr_of_BS_antennas,2*nr_of_BS_antennas))
      a3_exp = torch.tile(torch.unsqueeze(torch.sum((receiver_precoder[:,0,:,:])**2,1),-1),(1,2*nr_of_BS_antennas,2*nr_of_BS_antennas))    
      #print(a3_exp.shape,a2_exp.shape,  a1_exp.shape, channel[:, 0,:, :].shape)
      temp = a1_exp*a2_exp*a3_exp*torch.matmul(channel[:,0,:,:],channel[:,0,:,:].permute(0,2,1))
    
      # Next iterations
      for i in range(1, nr_of_users):
          a1_exp = torch.tile(torch.unsqueeze(mse_weights[:,i,:],-1),(1,2*nr_of_BS_antennas,2*nr_of_BS_antennas))
          a2_exp = torch.tile(torch.unsqueeze(user_weights[:,i,:],-1),[1,2*nr_of_BS_antennas,2*nr_of_BS_antennas])
          a3_exp = torch.tile(torch.unsqueeze(torch.sum((receiver_precoder[:,i,:,:])**2, 1),-1),(1,2*nr_of_BS_antennas,2*nr_of_BS_antennas))
          temp = temp + a1_exp*a2_exp*a3_exp*torch.matmul(channel[:,i,:,:],channel[:,i,:,:].permute(0,2,1))

      sum_gradient = temp 

      gradient = []

      # Gradient computation
      for i in range(nr_of_users):
        a1_exp = torch.tile(torch.unsqueeze(mse_weights[:,i,:],-1),(1,2*nr_of_BS_antennas,1)).float()
        a2_exp = torch.tile(torch.unsqueeze(user_weights[:,i,:],-1),[1,2*nr_of_BS_antennas,1]).float()
        
        gradient.append(step_size[step_i] * (-2.0*a1_exp*a2_exp*torch.matmul(channel[:,i,:,:].float(),receiver_precoder[:,i,:,:].float())+ 2*torch.matmul(sum_gradient.float(),initial_transmitter_precoder[:,i,:,:].float()))) 
      
      torch.stack(gradient)
      gradient = torch.stack(gradient).permute(1,0,2,3)
      output_temp = initial_transmitter_precoder - gradient

      output = []
      nr_of_samples_per_batch = channel_input.shape[0]
      for i in range(nr_of_samples_per_batch):
        if torch.linalg.norm(output_temp[i])**2 < total_power:
          output.append(output_temp[i])  
        else:
          output.append(np.sqrt(total_power)*output_temp[i]/torch.linalg.norm(output_temp[i]))

      initial_transmitter_precoder = torch.stack(output)

    # The WSR achieved with the transmitter precoder obtaiined at the current iteration is appended
    
    profit.append(compute_WSR_nn(user_weights, channel_input, initial_transmitter_precoder, noise_power,nr_of_users))


  final_precoder = initial_transmitter_precoder # this is the last transmitter precoder, i.e. the one that will be actually used for transmission
  WSR = torch.sum(torch.stack(profit)) # this is the cost function to maximize, i.e. the WSR obtained if we use the transmitter precoder that we have at each round of the loop 
  WSR_final = compute_WSR_nn(user_weights, channel_input, final_precoder, noise_power,nr_of_users)/nr_of_samples_per_batch # this is the WSR computed using the "final_precoder"

  return WSR, WSR_final



if __name__  == "__main__":
  WSR_WMMSE =[] # to store the WSR attained by the WMMSE
  WSR_ZF = [] # to store the WSR attained by the zero-forcing 
  WSR_RZF = [] # to store the WSR attained by the regularized zero-forcing
  WSR_nn = [] # to store the WSR attained by the deep unfolded WMMSE
  training_loss = []

  from net import Net
  from net import MMSE_Net
  net = Net([1.0 for i in range(4)])
  #mmse_net = MMSE_Net()
  optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
  print("start of session")
  start_of_time = time.time()
  all_step = []
  import scipy.io
  mat = scipy.io.loadmat('20.mat')
  H = mat["Hnoisy"]
  try:
    from tqdm import tqdm
    for i in tqdm(range(nr_of_batches_training)):
      batch_for_training = []
      initial_transmitter_precoder_batch = []
      mmse_net_input = []
      mmse_net_target = []
      # Building a batch for training
      batch_size = 100
      for ii in range(batch_size):
        channel_realization_nn, init_transmitter_precoder, W,_ = compute_channel(nr_of_BS_antennas, nr_of_users, total_power,np.array([[1]]), path_loss_option, path_loss_range)
        batch_for_training.append(channel_realization_nn)
        initial_transmitter_precoder_batch.append(init_transmitter_precoder)
        if ii == 0:
          target,_,_,_= run_WMMSE(epsilon, W, scheduled_users, total_power, noise_power, user_weights_for_regular_WMMSE, nr_of_iterations-1, log = False)
          mmse_net_target.append(target)
        
        #print(np.array(init_transmitter_precoder.shape, init_transmitter_precoder)
        #assert 0
      # Training
      channel_input = torch.as_tensor(np.array(batch_for_training), dtype=torch.float32)
      initial_tp = torch.as_tensor(np.array(initial_transmitter_precoder_batch), dtype=torch.float32)       
      #assert 0
      step  =net()
      
      step = step[0].float()
      #step = torch.as_tensor(step, dtype=torch.float32)
      step_size_1, step_size_2, step_size_3, step_size_4 = step[0], step[1], step[2], step[3]
      all_step.append([step_size_1, step_size_2, step_size_3, step_size_4])
      all_step.append([step_size_1, step_size_2, step_size_3, step_size_4])

      obj, obj_final = calc_wsr(channel_input, initial_tp, step)
      print(obj_final.item()) 
      
      training_loss.append(-1 * obj.item())
      obj = -obj
      optimizer.zero_grad()
      obj.backward()
      optimizer.step()
  except KeyboardInterrupt:
    path = './net_3.pth'
    exit()
  path = './net_3.pth'
  matrix_file = ''
  torch.save(net.state_dict(), path)#exit()

  print("step size", all_step)
  print("Training took:", time.time()-start_of_time)

  # For repeatability
  np.random.seed(1234)
  import scipy.io
  mat = scipy.io.loadmat('20.mat')
  H = mat["Hnoisy"]
  print(H.shape)
  for i in range(nr_of_batches_test):    
    batch_for_testing = []
    initial_transmitter_precoder_batch = []
    WSR_WMMSE_batch = 0.0
    WSR_ZF_batch = 0.0
    WSR_RZF_batch = 0.0

    # Building a batch for testing

    for ii in range(20):#nr_of_samples_per_batch):       
      channel_realization_nn, init_transmitter_precoder, channel_realization_regular, regularization_parameter_for_RZF_solution = compute_channel(nr_of_BS_antennas, nr_of_users, total_power, H[:, :, ii], path_loss_option, path_loss_range)    
      
      #assert 0
      # Compute the WMMSE solution
      _,_,_,WSR_WMMSE_one_sample = run_WMMSE(epsilon, channel_realization_regular, scheduled_users, total_power, noise_power, user_weights_for_regular_WMMSE, nr_of_iterations-1, log = False)      
      WSR_WMMSE_batch =  WSR_WMMSE_batch + WSR_WMMSE_one_sample
      print("WMMSE ", WSR_WMMSE_one_sample)
      
      # Compute the zero-forcing solution
      ZF_solution = zero_forcing(channel_realization_regular, total_power)
      WSR_ZF_one_sample = compute_weighted_sum_rate(user_weights_for_regular_WMMSE , channel_realization_regular, ZF_solution, noise_power, scheduled_users)
      WSR_ZF_batch =  WSR_ZF_batch + WSR_ZF_one_sample
      
      # Compute the regilarized zero-forcing solution
      RZF_solution = regularized_zero_forcing(channel_realization_regular, total_power, regularization_parameter_for_RZF_solution, path_loss_option)
      WSR_RZF_one_sample = compute_weighted_sum_rate(user_weights_for_regular_WMMSE , channel_realization_regular, RZF_solution, noise_power, scheduled_users)
      WSR_RZF_batch =  WSR_RZF_batch + WSR_RZF_one_sample

      batch_for_testing.append(channel_realization_nn)
      initial_transmitter_precoder_batch.append(init_transmitter_precoder)
    #print("begin test!") 
    #Testing
    channel_input = torch.as_tensor(batch_for_testing)
    initial_tp = torch.as_tensor(initial_transmitter_precoder_batch)
    #net.load_state_dict(torch.load('./net_1.pth', map_location=torch.device("cuda:0")))
    
    step = net()
    step = step[0]
    #step_size_1, step_size_2, step_size_3, step_size_4 = step[0], step[1], step[2], step[3] 
    print(channel_input.shape, initial_tp.shape)
    _, obj_final = calc_wsr(channel_input, initial_tp, step)
    WSR_nn.append(obj_final.item())
    WSR_WMMSE.append(WSR_WMMSE_batch/20)
    WSR_ZF.append(WSR_ZF_batch/20)
    WSR_RZF.append(WSR_RZF_batch/20)   
        
  print("Training and testing took:", time.time()-start_of_time)
  print("The WSR acheived with the deep unfolded WMMSE algorithm is: ",np.mean(WSR_nn))
  print(WSR_nn)
  print("The WSR acheived with the WMMSE algorithm is: ",np.mean(WSR_WMMSE))
  print(WSR_WMMSE)
  print("The WSR acheived with the zero forcing is: ",np.mean(WSR_ZF))
  print("The WSR acheived with the regularized zero forcing is: ",np.mean(WSR_RZF))

  #path = './net_3.pth'
  #matrix_file = ''
  #torch.save(net.state_dict(), path)


  #plt.figure()
  #plt.plot(training_loss)
  #plt.ylabel("Training loss")
  #plt.xlabel("Sample index")    
  #plt.savefig("training_loss.png")



