import torch as th# Import libraries
import torch as th
import numpy as np
import copy
from copy import deepcopy
import time
import matplotlib.pyplot as plt

class channel:
  def __init__(self, nr_of_BS_antennas, nr_of_users, total_power, path_loss_option, path_loss_range):
    self.state = None
    self.nr_of_BS_antennas = nr_of_BS_antennas
    self.nr_of_users =  nr_of_users
    self.total_power = total_power
    self.path_loss_option = path_loss_option
    self.path_loss_range = path_loss_range
    self.max_step = 1 
  def reset(self, ):
    channel_nn, initial_transmitter_precoder,_,_ = self.compute_channel(self.nr_of_BS_antennas, self.nr_of_users, self.total_power, \
                                                                    self.path_loss_option, self.path_loss_range)
    self.channel_nn = th.as_tensor(np.array(channel_nn))
    self.initial_transmitter_precoder = th.as_tensor(np.array(initial_transmitter_precoder))
    self.state = th.cat((th.as_tensor(self.channel_nn), th.as_tensor(self.initial_transmitter_precoder)), dim = 2).flatten()
    return self.state
  def step(self, action):

    #print(self.channel_nn.unsqueeze(0).shape) 
    reward, _ = self.calc_wsr(self.channel_nn.unsqueeze(0), self.initial_transmitter_precoder.unsqueeze(0), action[0], action[1], action[2], action[3])
    #print(reward.shape)
    return None, reward, True, {}
    
    
    
    
# Computes a channel realization and returns it in two formats, one for the WMMSE and one for the deep unfolded WMMSE.
# It also returns the initialization value of the transmitter precoder, which is used as input in the computation graph of the deep unfolded WMMSE.
  def compute_channel(self, nr_of_BS_antennas, nr_of_users, total_power, path_loss_option = False, path_loss_range = [-5,5] ):
    channel_nn = []
    initial_transmitter_precoder = []
    channel_WMMSE = np.zeros((nr_of_users, nr_of_BS_antennas)) + 1j*np.zeros((nr_of_users, nr_of_BS_antennas))

    
    for i in range(nr_of_users):

        regularization_parameter_for_RZF_solution = 0
        path_loss = 0 # path loss is 0 dB by default, otherwise it is drawn randomly from a uniform distribution (N.B. it is different for each user)
        if path_loss_option == True:
          path_loss = np.random.uniform(path_loss_range[0],path_loss_range[-1])
          regularization_parameter_for_RZF_solution = regularization_parameter_for_RZF_solution + 1/((10**(path_loss/10))*total_power) # computed as in "MMSE precoding for multiuser MISO downlink transmission with non-homogeneous user SNR conditions" by D.H. Nguyen and T. Le-Ngoc

        result_real = np.sqrt(10**(path_loss/10))*np.sqrt(0.5)*np.random.normal(size = (nr_of_BS_antennas,1))
        result_imag  =  np.sqrt(10**(path_loss/10))*np.sqrt(0.5)*np.random.normal(size = (nr_of_BS_antennas,1))
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

    return channel_nn, initial_transmitter_precoder, channel_WMMSE, regularization_parameter_for_RZF_solution


  def calc_wsr(self, channel_input, initial_tp, step_size_1, step_size_2, step_size_3, step_size_4):
    initial_transmitter_precoder = initial_tp

    # The number of step sizes depends on the selected number of PGD layers, the number of elements for each step size initializer depends on the selected number of deep unfolded iterations
    profit = [] # stores the WSR obtained at each iteration
    for loop in range(0,self.nr_of_iterations_nn):
      user_interference2 = []
      
      for batch_index in range(self.nr_of_samples_per_batch): 
        if batch_index >= channel_input.shape[0]:
          break
        user_interference_single = []
        for i in range(self.nr_of_users):
          temp = 0.0
          for j in range(self.nr_of_users):
            temp = temp + th.sum((th.matmul(th.t(channel_input[batch_index, i,:,:]),initial_transmitter_precoder[batch_index,j,:,:]))**2)
          user_interference_single.append(temp + self.noise_power)
        user_interference2.append(th.as_tensor(user_interference_single).unsqueeze(0))
      user_interference2 = th.cat(user_interference2)

      user_interference_exp2 = th.tile(th.unsqueeze(th.tile(th.unsqueeze(user_interference2,-1),(1,1,2)),-1),(1,1,1,1))
      receiver_precoder_temp = (th.matmul(channel_input.permute(0,1,3,2),initial_transmitter_precoder))
      # Optimize the receiver precoder 
      receiver_precoder = th.divide(receiver_precoder_temp,user_interference_exp2)

      # Optimize the mmse weights 
      self_interference = th.sum((th.matmul(channel_input.permute(0,1,3,2),initial_transmitter_precoder))**2, 2)

      inter_user_interference_total = []

      for batch_index in range(self.nr_of_samples_per_batch):
        if batch_index >= channel_input.shape[0]:
          break      
        inter_user_interference_temp = []
        for i in range(self.nr_of_users):
          temp = 0.0
          for j in range(self.nr_of_users):
            if j != i:
              temp = temp + th.sum((th.matmul(th.t(channel_input[batch_index, i,:,:]),initial_transmitter_precoder[batch_index,j,:,:]))**2)
          inter_user_interference_temp.append(temp + self.noise_power) # $sum{|(h_i)*H,v_i}|**2 + noise_power$
        inter_user_interference = th.as_tensor(inter_user_interference_temp).unsqueeze(0).reshape((1,self.nr_of_users,1)) # Nx1 $sum{|(h_i)*H,v_i}|**2 + noise_power$
        inter_user_interference_total.append(inter_user_interference)
      inter_user_interference_total = th.cat(inter_user_interference_total)
          
      mse_weights = (th.divide(self_interference,inter_user_interference_total)) + 1.0
      
      # Optimize the transmitter precoder through PGD
      #################################################################

      channel = channel_input
      # First iteration
      a1_exp = th.tile(th.unsqueeze(mse_weights[:,0,:],-1),(1,2*self.nr_of_BS_antennas,2*self.nr_of_BS_antennas))
      a2_exp = th.tile(th.unsqueeze(self.user_weights[:,0,:],-1),(1,2*self.nr_of_BS_antennas,2*self.nr_of_BS_antennas))
      a3_exp = th.tile(th.unsqueeze(th.sum((receiver_precoder[:,0,:,:])**2,1),-1),(1,2*self.nr_of_BS_antennas,2*self.nr_of_BS_antennas))    
      temp = a1_exp*a2_exp*a3_exp*th.matmul(channel[:,0,:,:],channel[:,0,:,:].permute(0,2,1))
      
      # Next iterations
      for i in range(1, self.nr_of_users):
        a1_exp = th.tile(th.unsqueeze(mse_weights[:,i,:],-1),(1,2*self.nr_of_BS_antennas,2*self.nr_of_BS_antennas))
        a2_exp = th.tile(th.unsqueeze(th.sum((receiver_precoder[:,i,:,:])**2, 1),-1),(1,2*self.nr_of_BS_antennas,2*self.nr_of_BS_antennas))
        temp = temp + a1_exp*a2_exp*a3_exp*th.matmul(channel[:,i,:,:],channel[:,i,:,:].permute(0,2,1))

      sum_gradient = temp 

      gradient = []

      # Gradient computation
      for i in range(self.nr_of_users):
        a1_exp = th.tile(th.unsqueeze(mse_weights[:,i,:],-1),(1,2*self.nr_of_BS_antennas,1))
        a2_exp = th.tile(th.unsqueeze(self.user_weights[:,i,:],-1),[1,2*self.nr_of_BS_antennas,1])
        gradient.append(step_size_1 * (-2.0*a1_exp*a2_exp*th.matmul(channel[:,i,:,:],receiver_precoder[:,i,:,:])+ 2*th.matmul(sum_gradient,initial_transmitter_precoder[:,i,:,:]))) 
        
      th.stack(gradient)
      gradient = th.stack(gradient).permute(1,0,2,3)
      output_temp = initial_transmitter_precoder - gradient

      output = []
      for i in range(self.nr_of_samples_per_batch):
        if th.linalg.norm(output_temp[i])**2 < self.total_power:
          output.append(output_temp[i])  
        else:
          output.append(np.sqrt(self.total_power)*output_temp[i]/th.linalg.norm(output_temp[i]))
        

      transmitter_precoder1 = th.stack(output)

      ##############################################################################
      channel = channel_input
      initial_transmitter_precoder = transmitter_precoder1
      # First iteration
      a1_exp = th.tile(th.unsqueeze(mse_weights[:,0,:],-1),(1,2*self.nr_of_BS_antennas,2*nr_of_BS_antennas))
      a2_exp = th.tile(th.unsqueeze(self.user_weights[:,0,:],-1),(1,2*self.nr_of_BS_antennas,2*nr_of_BS_antennas))
      a3_exp = th.tile(th.unsqueeze(th.sum((receiver_precoder[:,0,:,:])**2,1),-1),(1,2*nr_of_BS_antennas,2*nr_of_BS_antennas))    
      temp = a1_exp*a2_exp*a3_exp*th.matmul(channel[:,0,:,:],channel[:,0,:,:].permute(0,2,1))
      
      # Next iterations
      for i in range(1, nr_of_users):
        a1_exp = th.tile(th.unsqueeze(mse_weights[:,i,:],-1),(1,2*nr_of_BS_antennas,2*nr_of_BS_antennas))
        a2_exp = th.tile(th.unsqueeze(user_weights[:,i,:],-1),[1,2*nr_of_BS_antennas,2*nr_of_BS_antennas])
        a3_exp = th.tile(th.unsqueeze(th.sum((receiver_precoder[:,i,:,:])**2, 1),-1),(1,2*nr_of_BS_antennas,2*nr_of_BS_antennas))
        temp = temp + a1_exp*a2_exp*a3_exp*th.matmul(channel[:,i,:,:],channel[:,i,:,:].permute(0,2,1))

      sum_gradient = temp 

      gradient = []

      # Gradient computation
      for i in range(nr_of_users):
        a1_exp = th.tile(th.unsqueeze(mse_weights[:,i,:],-1),(1,2*nr_of_BS_antennas,1))
        a2_exp = th.tile(th.unsqueeze(user_weights[:,i,:],-1),[1,2*nr_of_BS_antennas,1])
        gradient.append(step_size_2 * (-2.0*a1_exp*a2_exp*th.matmul(channel[:,i,:,:],receiver_precoder[:,i,:,:])+ 2*th.matmul(sum_gradient,initial_transmitter_precoder[:,i,:,:]))) 
        
      th.stack(gradient)
      gradient = th.stack(gradient).permute(1,0,2,3)
      output_temp = initial_transmitter_precoder - gradient

      output = []
      for i in range(nr_of_samples_per_batch):
        if th.linalg.norm(output_temp[i])**2 < total_power:
          output.append(output_temp[i])  
        else:
          output.append(np.sqrt(total_power)*output_temp[i]/th.linalg.norm(output_temp[i]))
        

      transmitter_precoder2 = th.stack(output)
      
      ##############################################################################
      
      
      channel = channel_input
      initial_transmitter_precoder = transmitter_precoder2

      # First iteration
      a1_exp = th.tile(th.unsqueeze(mse_weights[:,0,:],-1),(1,2*nr_of_BS_antennas,2*nr_of_BS_antennas))
      a2_exp = th.tile(th.unsqueeze(user_weights[:,0,:],-1),(1,2*nr_of_BS_antennas,2*nr_of_BS_antennas))
      a3_exp = th.tile(th.unsqueeze(th.sum((receiver_precoder[:,0,:,:])**2,1),-1),(1,2*nr_of_BS_antennas,2*nr_of_BS_antennas))    
      temp = a1_exp*a2_exp*a3_exp*th.matmul(channel[:,0,:,:],channel[:,0,:,:].permute(0,2,1))
      
      # Next iterations
      for i in range(1, nr_of_users):
        a1_exp = th.tile(th.unsqueeze(mse_weights[:,i,:],-1),(1,2*nr_of_BS_antennas,2*nr_of_BS_antennas))
        a2_exp = th.tile(th.unsqueeze(user_weights[:,i,:],-1),[1,2*nr_of_BS_antennas,2*nr_of_BS_antennas])
        a3_exp = th.tile(th.unsqueeze(th.sum((receiver_precoder[:,i,:,:])**2, 1),-1),(1,2*nr_of_BS_antennas,2*nr_of_BS_antennas))
        temp = temp + a1_exp*a2_exp*a3_exp*th.matmul(channel[:,i,:,:],channel[:,i,:,:].permute(0,2,1))

      sum_gradient = temp 

      gradient = []

      # Gradient computation
      for i in range(nr_of_users):
        a1_exp = th.tile(th.unsqueeze(mse_weights[:,i,:],-1),(1,2*nr_of_BS_antennas,1))
        a2_exp = th.tile(th.unsqueeze(user_weights[:,i,:],-1),[1,2*nr_of_BS_antennas,1])
        gradient.append(step_size_3 * (-2.0*a1_exp*a2_exp*th.matmul(channel[:,i,:,:],receiver_precoder[:,i,:,:])+ 2*th.matmul(sum_gradient,initial_transmitter_precoder[:,i,:,:]))) 
        
      th.stack(gradient)
      gradient = th.stack(gradient).permute(1,0,2,3)
      output_temp = initial_transmitter_precoder - gradient

      output = []
      for i in range(nr_of_samples_per_batch):
        if th.linalg.norm(output_temp[i])**2 < total_power:
          output.append(output_temp[i])  
        else:
          output.append(np.sqrt(total_power)*output_temp[i]/th.linalg.norm(output_temp[i]))
        

      transmitter_precoder3 = th.stack(output)
    ######################################################################################
      channel = channel_input
      initial_transmitter_precoder = transmitter_precoder3
      
      # First iteration
      a1_exp = th.tile(th.unsqueeze(mse_weights[:,0,:],-1),(1,2*nr_of_BS_antennas,2*nr_of_BS_antennas))
      a2_exp = th.tile(th.unsqueeze(user_weights[:,0,:],-1),(1,2*nr_of_BS_antennas,2*nr_of_BS_antennas))
      a3_exp = th.tile(th.unsqueeze(th.sum((receiver_precoder[:,0,:,:])**2,1),-1),(1,2*nr_of_BS_antennas,2*nr_of_BS_antennas))    
      temp = a1_exp*a2_exp*a3_exp*th.matmul(channel[:,0,:,:], channel[:,0,:,:].permute(0,2,1))
      
      # Next iterations
      for i in range(1, nr_of_users):
        a1_exp = th.tile(th.unsqueeze(mse_weights[:,i,:],-1),(1,2*nr_of_BS_antennas,2*nr_of_BS_antennas))
        a2_exp = th.tile(th.unsqueeze(user_weights[:,i,:],-1),[1,2*nr_of_BS_antennas,2*nr_of_BS_antennas])
        a3_exp = th.tile(th.unsqueeze(th.sum((receiver_precoder[:,i,:,:])**2, 1),-1),(1,2*nr_of_BS_antennas,2*nr_of_BS_antennas))
        temp = temp + a1_exp*a2_exp*a3_exp*th.matmul(channel[:,i,:,:], channel[:,i,:,:].permute(0,2,1))

      sum_gradient = temp 

      gradient = []

      # Gradient computation
      for i in range(nr_of_users):
        a1_exp = th.tile(th.unsqueeze(mse_weights[:,i,:],-1),(1,2*nr_of_BS_antennas,1))
        a2_exp = th.tile(th.unsqueeze(user_weights[:,i,:],-1),[1,2*nr_of_BS_antennas,1])
        gradient.append(step_size_4 * (-2.0*a1_exp*a2_exp*th.matmul(channel[:,i,:,:],receiver_precoder[:,i,:,:])+ 2*th.matmul(sum_gradient,initial_transmitter_precoder[:,i,:,:]))) 
        
      th.stack(gradient)
      gradient = th.stack(gradient).permute(1,0,2,3)
      output_temp = initial_transmitter_precoder - gradient

      output = []
      for i in range(nr_of_samples_per_batch):
        if th.linalg.norm(output_temp[i])**2 < total_power:
          output.append(output_temp[i])  
        else:
          output.append(np.sqrt(total_power)*output_temp[i]/th.linalg.norm(output_temp[i]))
        

      transmitter_precoder = th.stack(output)
    ######################################################################################
      
      initial_transmitter_precoder = transmitter_precoder



      # The WSR achieved with the transmitter precoder obtained at the current iteration is appended
      profit.append(compute_WSR_nn(user_weights, channel_input, initial_transmitter_precoder, noise_power,nr_of_users))


    final_precoder = initial_transmitter_precoder # this is the last transmitter precoder, i.e. the one that will be actually used for transmission

    WSR = th.sum(th.stack(profit)) # this is the cost function to maximize, i.e. the WSR obtained if we use the transmitter precoder that we have at each round of the loop 
    WSR_final = compute_WSR_nn(user_weights, channel_input, final_precoder, noise_power,nr_of_users)/nr_of_samples_per_batch # this is the WSR computed using the "final_precoder"

    return WSR, WSR_final


      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
