"""Main VRP solver model using attention mechanism."""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from layers import GraphEmbedding, AttentionModule, Glimpse, Pointer


class AttentionVRP(nn.Module):
    """VRP solver using self-attention and pointer networks."""
    
    def __init__(self, embedding_size, hidden_size, seq_len, n_head=4, C=10):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.n_head = n_head
        self.C = C
        
        # Node embedding
        self.embedding = GraphEmbedding(2, embedding_size)
        
        # Self-attention encoder (Pre-LN inside)
        self.encoder = AttentionModule(embedding_size, n_head)
        
        # Context embedding
        self.h_context_embed = nn.Linear(embedding_size, embedding_size)
        self.v_weight_embed = nn.Linear(embedding_size * 2, embedding_size)
        
        # Glimpse and pointer for decoding
        self.glimpse = Glimpse(embedding_size, hidden_size, n_head)
        self.pointer = Pointer(embedding_size, hidden_size, 1, C)
        
        # Learnable initial query vector
        self.init_w = nn.Parameter(torch.Tensor(2 * embedding_size))
        self.init_w.data.uniform_(-1, 1)
        
    def forward(self, inputs):
        """
        Args:
            inputs: FloatTensor [batch_size, seq_len, 2]
            
        Returns:
            log_probs: log probabilities of selected actions [batch_size, seq_len]
            actions: selected node indices [batch_size, seq_len]
        """
        batch_size = inputs.shape[0]
        
        # Encode all nodes (can run under AMP)
        embedded = self.embedding(inputs)  # [batch_size, seq_len, embedding_size]
        encoded = self.encoder(embedded)   # [batch_size, seq_len, embedding_size]
        
        # Compute context
        h_mean = encoded.mean(dim=1)  # [batch_size, embedding_size]
        h_context = self.h_context_embed(h_mean)
        
        # Initialize query with learned weights (broadcasting to batch)
        query = h_context + self.v_weight_embed(self.init_w)  # [batch_size, embedding_size]
        
        # Decode path
        log_probs = []
        actions = []
        mask = torch.zeros(batch_size, self.seq_len, dtype=torch.bool, device=inputs.device)
        first_node_h = None
        
        for step in range(self.seq_len):
            # Glimpse to refine query
            _, glimpsed_query = self.glimpse(query, encoded, mask)
            
            # Point to next node - get logits (FP32)
            logits = self.pointer(glimpsed_query, encoded, mask)  # [batch_size, seq_len], FP32
            
            # Use logits directly for more stable sampling (already in FP32 from Pointer)
            categorical = Categorical(logits=logits)
            action = categorical.sample()  # [batch_size]
            
            # Store log probability and action
            log_probs.append(categorical.log_prob(action))  # [batch_size]
            actions.append(action)  # [batch_size]
            
            # Update mask
            mask = mask.scatter(1, action.unsqueeze(1), True)
            
            # Gather selected node embedding
            action_expanded = action.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.embedding_size)
            selected_h = encoded.gather(1, action_expanded).squeeze(1)  # [batch_size, embedding_size]
            
            if first_node_h is None:
                first_node_h = selected_h
                
            # Update query for next step
            query = h_context + self.v_weight_embed(torch.cat([first_node_h, selected_h], dim=-1))
        
        # Stack results
        log_probs = torch.stack(log_probs, dim=1)  # [batch_size, seq_len]
        actions = torch.stack(actions, dim=1)       # [batch_size, seq_len]
        
        return log_probs, actions


class VRPSolver(nn.Module):
    """Wrapper for VRP solver with reward calculation."""
    
    def __init__(self, embedding_size, hidden_size, seq_len, n_head=4, C=10):
        super().__init__()
        self.model = AttentionVRP(embedding_size, hidden_size, seq_len, n_head, C)
        
    def forward(self, inputs):
        """
        Args:
            inputs: FloatTensor [batch_size, seq_len, 2]
            
        Returns:
            rewards: tour lengths [batch_size]
            log_probs: log probabilities [batch_size, seq_len]
            actions: selected actions [batch_size, seq_len]
        """
        log_probs, actions = self.model(inputs)
        
        # Gather nodes in tour order
        batch_size = inputs.size(0)
        tour = inputs.gather(1, actions.unsqueeze(2).expand(-1, -1, 2))  # [batch_size, seq_len, 2]
        
        # Calculate tour length
        rewards = self.calculate_reward(tour)
        
        return rewards, log_probs, actions
    
    @staticmethod
    def calculate_reward(tour):
        """Calculate total tour length.
        
        Args:
            tour: FloatTensor [batch_size, seq_len, 2]
            
        Returns:
            lengths: FloatTensor [batch_size]
        """
        # Calculate distances between consecutive nodes
        distances = torch.norm(tour[:, 1:] - tour[:, :-1], dim=2)
        
        # Add distance from last node back to first
        distances_to_first = torch.norm(tour[:, -1] - tour[:, 0], dim=1)
        
        # Total tour length
        total_length = distances.sum(dim=1) + distances_to_first
        
        return total_length