import torch
from  rlsolver.methods.iSCO.config.tsp_config import *
from torch.func import vmap
from  rlsolver.methods.iSCO.util import math_util
import torch.nn.functional as F


class iSCO:
    def __init__(self,params_dict):
        self.batch_size = BATCH_SIZE
        self.device = DEVICE
        self.chain_length = CHAIN_LENGTH
        self.init_temperature = torch.tensor(INIT_TEMPERATURE,device=self.device)
        self.final_temperature = torch.tensor(FINAL_TEMPERATURE,device=self.device)
        self.num_nodes = params_dict['num_nodes']
        self.distance = params_dict['distance']
        self.nearest_indices = params_dict['nearest_indices']
        self.farthest_indices = params_dict['farthest_indices']

    def random_gen_init_sample(self,params_dict):

        sample = torch.arange(0, params_dict['num_nodes'],device=self.device,dtype=torch.long)[torch.randperm(params_dict['num_nodes'])]
    
        return sample
    
    def step(self,x,path_length,temperature):
        traj = torch.empty((path_length+1,self.num_nodes),dtype=torch.float,device=self.device)
        delta_xy = torch.empty((path_length,self.num_nodes),dtype=torch.float,device=self.device)
        mask_list = torch.empty((path_length,),dtype=torch.long,device=self.device)
        cur_x = x.clone()

        for i in range(path_length):
            cur_x,delta_x2y,ll_x2y,mask = self.proposal(cur_x,temperature)
            traj[i]=ll_x2y
            delta_xy[i] = delta_x2y
            mask_list[i] = mask
            if i == path_length-1:
                _,___,ll_x2y,__ = self.proposal(cur_x,temperature)
                traj[i+1] = ll_x2y

        ll_x2y = torch.sum(traj[0:-1][torch.arange(path_length,dtype=torch.long,device=self.device),mask_list])
        ll_y2x = torch.sum(traj[1:][torch.arange(path_length,dtype=torch.long,device=self.device),mask_list])
        delta_xy = torch.sum(delta_xy[torch.arange(path_length,dtype=torch.long,device=self.device),mask_list])

        log_acc = delta_xy + ll_y2x  - ll_x2y
        y,accetped = self.select_sample(log_acc, x, cur_x )
        
        return y,log_acc.exp()

        

    def proposal(self,sample,temperature):
        x = sample.clone()
        logratio,log_prob,indices = self.get_local_dist(x,temperature)
        mask =torch.multinomial(log_prob.exp(),1)
        temp = x[mask+1-self.num_nodes].clone()
        x[mask+1-self.num_nodes] = x[indices[mask]]
        x[indices[mask]] = temp
        return x,logratio,log_prob,mask
    
    
    def get_local_dist(self,sample,temperature):
        #log_prob是每点采样的权重，logratio是delta_yx
        x = sample.detach()
        mask = torch.arange(0,self.num_nodes,dtype=torch.long,device=self.device)
        logratio,indices = self.opt_2(x,mask,temperature)
        logits = self.apply_weight_function_logscale(logratio)
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
        return logratio,log_prob,indices    
    
    def opt_2(self,sample,mask,temperature):
        selected_mask = torch.where(torch.rand((self.num_nodes),device=self.device)<torch.tensor(K/(K+1),device=self.device),
                                    self.nearest_indices[sample][mask, torch.randint(0, K, (self.num_nodes,),device=self.device)],
                                    self.farthest_indices[sample][mask, torch.randint(0, self.num_nodes-K-1, (self.num_nodes,),device=self.device)])
        #找到每个点对应的selected_mask在sample中的索引位置
        sorted_t1, sorted_indices = torch.sort(sample)
        sorted_t2_indices = torch.searchsorted(sorted_t1, selected_mask)
        indices = sorted_indices[sorted_t2_indices]


        mask0 = mask-1
        mask1 = mask+1-self.num_nodes
        mask2 = mask+2-self.num_nodes
        mask3 = mask+3-self.num_nodes

        indnces1 = indices-1
        indnces2 = indices+1-self.num_nodes

        condition1 = (mask + 1) % self.num_nodes == (indices[mask] - 1) % 100  
        condition2 = mask == indices[mask]                                

        # 合并条件
        combined_condition = condition1 | condition2 

        delta_y = torch.where(combined_condition,torch.where(condition1,self.distance[sample,sample[mask1]]+self.distance[sample[mask3],sample[mask2]]
                                                            -(self.distance[sample,sample[mask2]]+self.distance[sample[mask1],sample[mask3]]),
                                                            self.distance[sample,sample[mask0]]+self.distance[sample[mask1],sample[mask2]]
                                                            -(self.distance[sample,sample[mask2]]+self.distance[sample[mask1],sample[mask0]]))
                            ,(self.distance[sample,sample[mask1]]
                    +self.distance[sample[mask1],sample[mask2]]
                    +self.distance[selected_mask,sample[indnces1]]
                    +self.distance[selected_mask,sample[indnces2]]
                    -(self.distance[selected_mask,sample[mask]]
                        +self.distance[selected_mask,sample[mask2]]
                        +self.distance[sample[mask1],sample[indnces1]]
                        +self.distance[sample[mask1],sample[indnces2]])))
        return delta_y/(temperature*100),indices

    
    def select_sample(self,log_acc, x, y):
        y, accepted = math_util.mh_step(log_acc, x, y )
        return y,accepted
    
    def apply_weight_function_logscale(self,logratio):
        logits = logratio/2
        # logits = th.nn.functional.logsigmoid(logratio)
        return logits