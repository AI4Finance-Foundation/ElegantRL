"""Main TSP solver model using attention mechanism with vmap-based forward pass."""

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.func import vmap

from layers import GraphEmbedding, AttentionModule, Glimpse, Pointer


class AttentionTSP(nn.Module):
    """TSP solver using self-attention and pointer networks."""
    
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
    
    def compute_glimpse_and_logits(self, query, encoded, mask):
        """Compute glimpse and logits for a batch outside of vmap.
        
        Args:
            query: [batch_size, embedding_size]
            encoded: [batch_size, seq_len, embedding_size]
            mask: [batch_size, seq_len] boolean mask
            
        Returns:
            glimpsed_query: [batch_size, embedding_size]
            logits: [batch_size, seq_len]
        """
        # Glimpse to refine query
        _, glimpsed_query = self.glimpse(query, encoded, mask)
        
        # Point to next node - get logits
        logits = self.pointer(glimpsed_query, encoded, mask)
        
        return glimpsed_query, logits
    
    def decode_step_single(self, logits, encoded, mask, h_context, first_node_h, random_uniform):
        """Single step of autoregressive decoding for one instance (no batch dim).
        
        Args:
            logits: [seq_len] pre-computed logits
            encoded: [seq_len, embedding_size]
            mask: [seq_len] boolean mask
            h_context: [embedding_size]
            first_node_h: [embedding_size]
            random_uniform: float in [0, 1) for sampling
            
        Returns:
            tuple: (log_prob, action, new_query, new_mask, new_first_node_h)
        """
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Sample action using provided random value (inverse transform sampling)
        cumsum = torch.cumsum(probs, dim=-1)
        action = torch.searchsorted(cumsum, random_uniform, right=False)
        action = torch.clamp(action, 0, self.seq_len - 1)  # Safety clamp
        
        # Calculate log probability using gather to avoid indexing issues in vmap
        action_expanded = action.unsqueeze(0)  # [1]
        selected_prob = torch.gather(probs, 0, action_expanded).squeeze(0)  # scalar
        log_prob = torch.log(selected_prob + 1e-10)  # Add small epsilon for numerical stability
        
        # Update mask (functional, no in-place)
        new_mask = mask.clone()
        new_mask = torch.scatter(new_mask, 0, action_expanded, torch.ones_like(action_expanded, dtype=torch.bool))
        
        # Gather selected node embedding using gather
        action_for_encoded = action_expanded.unsqueeze(0).expand(self.embedding_size, -1).t()  # [1, embedding_size]
        selected_h = torch.gather(encoded, 0, action_for_encoded).squeeze(0)  # [embedding_size]
        
        # Handle first_node_h
        is_first = (first_node_h == 0).all()  # Check if it's the zero tensor (uninitialized)
        new_first_node_h = torch.where(is_first.unsqueeze(0), selected_h, first_node_h)
        
        # Update query for next step
        concat_features = torch.cat([new_first_node_h, selected_h], dim=-1)  # [2*embedding_size]
        new_query = h_context + self.v_weight_embed(concat_features)  # [embedding_size]
        
        return log_prob, action, new_query, new_mask, new_first_node_h
    
    def forward(self, inputs):
        """
        Args:
            inputs: FloatTensor [batch_size, seq_len, 2]
            
        Returns:
            log_probs: log probabilities of selected actions [batch_size, seq_len]
            actions: selected node indices [batch_size, seq_len]
        """
        batch_size = inputs.shape[0]
        device = inputs.device
        
        # Encode all nodes (can run under AMP)
        embedded = self.embedding(inputs)  # [batch_size, seq_len, embedding_size]
        encoded = self.encoder(embedded)   # [batch_size, seq_len, embedding_size]
        
        # Compute context
        h_mean = encoded.mean(dim=1)  # [batch_size, embedding_size]
        h_context = self.h_context_embed(h_mean)  # [batch_size, embedding_size]
        
        # Initialize query with learned weights (broadcasting to batch)
        init_query = h_context + self.v_weight_embed(self.init_w)  # [batch_size, embedding_size]
        
        # Initialize states
        query = init_query
        mask = torch.zeros(batch_size, self.seq_len, dtype=torch.bool, device=device)
        # Initialize first_node_h with zeros (will be detected as uninitialized in decode_step_single)
        first_node_h = torch.zeros(batch_size, self.embedding_size, device=device)
        
        # Pre-allocate outputs
        log_probs = torch.zeros(batch_size, self.seq_len, device=device)
        actions = torch.zeros(batch_size, self.seq_len, dtype=torch.long, device=device)
        
        # Create vmapped decode function for sampling and state updates
        vmapped_decode_step = vmap(
            self.decode_step_single,
            in_dims=(0, 0, 0, 0, 0, 0),  # All inputs have batch dimension
            out_dims=(0, 0, 0, 0, 0)  # All outputs have batch dimension
        )
        
        # Decode path step by step
        for step in range(self.seq_len):
            # Compute glimpse and logits for the entire batch (outside vmap)
            _, logits = self.compute_glimpse_and_logits(query, encoded, mask)
            
            # Generate random values for sampling (one per instance in batch)
            random_uniforms = torch.rand(batch_size, device=device)
            
            # Apply vmapped decode step (only for sampling and state updates)
            step_log_probs, step_actions, query, mask, first_node_h = vmapped_decode_step(
                logits, encoded, mask, h_context, first_node_h, random_uniforms
            )
            
            # Store results
            log_probs[:, step] = step_log_probs
            actions[:, step] = step_actions
        
        return log_probs, actions


class TSPSolver(nn.Module):
    """Wrapper for TSP solver with reward calculation."""
    
    def __init__(self, embedding_size, hidden_size, seq_len, n_head=4, C=10):
        super().__init__()
        self.model = AttentionTSP(embedding_size, hidden_size, seq_len, n_head, C)
        
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