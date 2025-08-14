# layers.py
from math import sqrt
import torch
import torch.nn as nn


class GraphEmbedding(nn.Module):
    """Linear embedding for graph nodes."""
    
    def __init__(self, input_size, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Linear(input_size, embedding_size)

    def forward(self, inputs):
        return self.embedding(inputs)


class Glimpse(nn.Module):
    """Multi-head dot-product attention for glimpsing."""
    
    def __init__(self, input_size, hidden_size, n_head):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.single_dim = hidden_size // n_head
        self.scale = 1.0 / sqrt(self.single_dim)
        
        self.W_q = nn.Linear(input_size, hidden_size)
        self.W_k = nn.Linear(input_size, hidden_size)
        self.W_v = nn.Linear(input_size, hidden_size)
        self.W_out = nn.Linear(hidden_size, input_size)

    def forward(self, query, target, mask=None):
        """
        Args:
            query: FloatTensor [batch_size, input_size]
            target: FloatTensor [batch_size, seq_len, input_size]
            mask: BoolTensor [batch_size, seq_len]
        
        Returns:
            alpha: attention weights [batch_size, n_head, seq_len]
            output: attended features [batch_size, input_size]
        """
        batch_size, seq_len, _ = target.shape

        # Compute Q, K, V (may be in AMP)
        q = self.W_q(query).reshape(batch_size, self.n_head, self.single_dim)
        k = self.W_k(target).reshape(batch_size, seq_len, self.n_head, self.single_dim)
        v = self.W_v(target).reshape(batch_size, seq_len, self.n_head, self.single_dim)
        
        # Rearrange for batched matrix multiplication
        k = k.permute(0, 2, 1, 3).contiguous()  # [batch_size, n_head, seq_len, single_dim]
        v = v.permute(0, 2, 1, 3).contiguous()  # [batch_size, n_head, seq_len, single_dim]
        
        # Compute attention scores in FP32 for numerical stability
        q_fp32 = q.float()
        k_fp32 = k.float()
        v_fp32 = v.float()
        
        scores = torch.einsum("bhd,bhld->bhl", q_fp32, k_fp32) * self.scale  # [batch_size, n_head, seq_len]
        
        # Apply mask with FP16-safe negative constant (-1e4 is within FP16 range)
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand(-1, self.n_head, -1)
            scores = scores.masked_fill(mask_expanded, -1e4)
            
            # Protection: handle fully masked rows (shouldn't happen but prevents NaN)
            all_masked = mask_expanded.all(dim=-1, keepdim=True)
            if all_masked.any():
                scores = scores.masked_fill(all_masked, 0.0)
        
        # Compute attention weights in FP32
        alpha = torch.softmax(scores, dim=-1)  # [batch_size, n_head, seq_len]
        
        # Apply attention to values
        attended = torch.einsum("bhl,bhld->bhd", alpha, v_fp32)  # [batch_size, n_head, single_dim]
        
        # Output projection (convert back to original dtype)
        output = self.W_out(attended.reshape(batch_size, -1).to(query.dtype))
        
        return alpha, output


class Pointer(nn.Module):
    """Pointer network for selecting next node (returns logits)."""
    
    def __init__(self, input_size, hidden_size, n_head=1, C=10):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.C = C
        
        self.W_q = nn.Linear(input_size, hidden_size)
        self.W_k = nn.Linear(input_size, hidden_size)
        # Note: Removed W_v to avoid unused parameters in DDP

    def forward(self, query, target, mask=None):
        """
        Args:
            query: FloatTensor [batch_size, input_size]
            target: FloatTensor [batch_size, seq_len, input_size]
            mask: BoolTensor [batch_size, seq_len]
            
        Returns:
            logits: unnormalized scores [batch_size, seq_len] in FP32
        """
        batch_size, seq_len, _ = target.shape
        
        q = self.W_q(query)  # [batch_size, hidden_size]
        k = self.W_k(target)  # [batch_size, seq_len, hidden_size]
        
        # Compute attention scores in FP32
        scores = torch.einsum("bh,blh->bl", q.float(), k.float())  # [batch_size, seq_len]
        logits = self.C * torch.tanh(scores)  # Bound to [-C, C], FP32
        
        # Apply mask with FP16-safe negative constant (-1e4 is within FP16 range)
        if mask is not None:
            logits = logits.masked_fill(mask, -1e4)
            
            # Protection: handle fully masked rows (shouldn't happen but prevents NaN)
            all_masked = mask.all(dim=-1)
            if all_masked.any():
                # Unmask last position for fully masked rows
                logits[all_masked, -1] = 0.0
        
        return logits  # FP32


class AttentionLayer(nn.Module):
    """Self-attention layer with Pre-LN and feed-forward network."""
    
    def __init__(self, embed_dim, n_heads, feed_forward_hidden=512, dropout=0.0):
        super().__init__()
        # Pre-LN architecture (more stable for deep networks)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_hidden),
            nn.ReLU(),
            nn.Linear(feed_forward_hidden, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Pre-LN Self-attention with residual connection
        normed = self.ln1(x)
        attn_output, _ = self.mha(normed, normed, normed)
        x = x + self.dropout(attn_output)
        
        # Pre-LN Feed-forward with residual connection
        normed = self.ln2(x)
        ffn_output = self.ffn(normed)
        x = x + self.dropout(ffn_output)
        
        return x


class AttentionModule(nn.Module):
    """Stack of self-attention layers."""
    
    def __init__(self, embed_dim, n_heads, feed_forward_hidden=512, n_layers=2, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionLayer(embed_dim, n_heads, feed_forward_hidden, dropout)
            for _ in range(n_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x