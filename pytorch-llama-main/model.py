import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # Number of heads for the Queries
    n_kv_heads: Optional[int] = None # Number of heads for K and V
    vocab_size: int = -1 # This will be set when we load the tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: 1e-5
    
    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048
    
    device: str = None
    
def precompute_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    assert head_dim % 2 == 0, "Dimension of head dim must be divided by 2"
    
    theta_numerator = torch.arange(0, head_dim, 2).float()
    
    # (head_dim /2)
    theta = 1.0 / (theta ** (theta_numerator) / head_dim).to(device)
    # (seq_len)
    m = torch.arange(seq_len, device=device)
    # (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()
    # (seq_len, head_dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex
    
def apply_rotary_embedding(x: torch.Tensor, freqs_compex: torch.Tensor, device: str):
    # [B, seq_len, H, Head_dim] -> [B, seq_len, H, Head_dim / 2, 2] -> [B, seq_len, H, Head_dim / 2]
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # [seq_len, head_dim / 2] -> [1, seq_len, 1, head_dim / 2]
    freqs_compex = freqs_compex.unsqueeze(0).unsqueeze(2)
    # [B, seq_len, H, Head_dim / 2] * [1, seq_len, 1, head_dim / 2] = [B, seq_len, H, Head_dim / 2]
    x_rotated = x_complex * freqs_compex
    # [B, seq_len, H, Head_dim / 2] -> [B, seq_len, H, Head_dim / 2, 2]
    x_out = torch.view_as_real(x_rotated)
    # [B, seq_len, H, Head_dim / 2, 2] -> [B, seq_len, H, Head_dim]
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

class RMSNorm(nn.Module):
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def _norm(self, x: torch.Tensor):
        # rsqrt = 1 / sqrt(x)
        # [B, seq_len, dim] * [B, seq_len, 1] = [B, seq_len, dim]
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True))

    def forward(self, x: torch.Tensor):
        # [dim] * [B, seq_len, dim] = [B, seq_len, dim]
        return self.weight * self._norm(x)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # (B, seq_len_KV, H_K, Head_Dim) -> (B, seq_len_KV, H_Q, Head_Dim)
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    return (
        # (B, seq_len_KV, H_K, 1, Head_Dim)
        x[:, :, :, None, :]
        # (B, seq_len_KV, H_K, N_rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        # (B, seq_len_KV, H_K * N_rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )

class FeedForward(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        hidden_dim = 4 * args.dim
        hidden_dim = int(hidden_dim * 2 / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        
        # hidden_dim / 256 + 1
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of -1) // args.multiple_of)
        
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
        
    def forward(self, x: torch.Tensor):
        # (B, seq_len, dim) -> (B, seq_len, hidden_dim)
        swish = F.silu(self.w1(x))
        # (B, seq_len, dim) -> (B, seq_len, hidden_dim)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) -> (B, Seq_Len, Dim)
        return self.w2(x)
        

class SelfAttention(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        # Number of the heads for the Keys and Values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is  None else args.n_kv_heads
        # Number of the heads for the Queries
        self.n_heads = args.n_heads
        # The ratio between Q heads and KV heads
        self.n_rep = self.n_heads // self.n_kv_heads       
        # The dimension of each head
        self.head_dim = args.dim // args.n_heads
        
        # (dim) -> (n_heads * head_dim)
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        # (dim) -> (n_kv_heads * head_dim) may smaller than (n_heads * head_dim)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # (n_kv_heads * head_dim) -> (dim)
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)
        
        # KV Cache
        # (B, seq_len, n_kv_heads, head_dim)
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, 1, dim)
        batch_size, seq_len, _ = x.shape
        
        # (B, 1, dim) -> (B, 1, H_Q * Head_Dim)
        xq = self.wq(x)
        # (B, 1 ,dim) -> (B, 1, H_K * Head_Dim)
        xk = self.wk(x)
        # (B, 1 ,dim) -> (B, 1, H_V * Head_Dim)
        xv = self.wv(x)
        
        # (B, 1, H_Q * Head_Dim) -> (B, 1, H_Q, Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        # (B, 1, H_K * Head_Dim) -> (B, 1, H_K * Head_Dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, 1, H_V * Head_Dim) -> (B, 1, H_V * Head_Dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # Apply Rope Positional Encoding to Q and K
        xq = apply_rotary_embedding(xq, freqs_complex, device=(x.device))
        xk = apply_rotary_embedding(xk, freqs_complex, device=(x.device))
        
        # Update the KV Cache
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv
        
        # Get the K and V
        # (B, seq_len_KV, H_K, Head_Dim)
        keys = self.cache_k[:batch_size, 0 : start_pos + seq_len]
        # (B, seq_len_KV, H_V, Head_Dim)
        values = self.cache_v[:batch_size, 0 : start_pos + seq_len]
        
        # 2 Q shares 1 same K and V
        
        # (B, seq_len_KV, H_K, Head_Dim) -> (B, seq_len_KV, H_Q, Head_Dim)
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)
        
        # Start attention 
        # (B, 1, H_Q, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        xq = xq.transpose(1, 2)
        # (B, seq_len_KV, H_Q, Head_Dim) -> (B, H_Q, seq_len_KV, Head_Dim)
        keys = keys.transpose(1, 2)
        # (B, seq_len_KV, H_Q, Head_Dim) -> (B, H_Q, seq_len_KV, Head_Dim)
        values = values.transpose(1, 2)
        
        # (B, H_Q, 1, Head_Dim) @ (B, H_Q, Head_Dim, seq_len_KV) -> (B, H_Q, 1, seq_len_KV)
        scores = xq @ keys.transpose(2, 3)
        
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        # (B, H_Q, 1, seq_len_KV) @ (B, H_Q, seq_len_KV, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        output = scores @ values
        # (B, H_Q, 1, Head_Dim) -> (B, 1, H_Q, Head_Dim) -> (B, 1, H_Q * Head_Dim) =(B, 1, Dim))
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output) #(B, 1, Dim) -> (B, 1, Dim)
        
                    

class EncoderBlock(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        self.dim = args.dim
        self.norm_eps = args.norm_eps
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        
        # Normalization BEFORE the attention block
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        # Normalization BEFORE the feed forward block
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
    
    def forward(self, x: torch.Tensor, start_pos: int ,freqs_complex: torch.Tensor):
        # [B, seq_len, dim] + [B, seq_len, dim] = [B, seq_len, dim] 
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        # [B, seq_len, dim] + [B, seq_len, dim] = [B, seq_len, dim]
        out = h + self.feed_forward.forward(self.ffn_norm(x))
        return out

class Transformer(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        assert args.vocab_size != -1, "Vocab size must be set!"
        
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)  # Token embedding layer
        
        self.layers = nn.ModuleList() # N EncoderBlock
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
            
        self.norm = RMSNorm(args.dim, eps=args.norm_eps) # RMSNorm
        self.output = nn.Linear(args.dim, args.vocab_size) # Output linear
        
        self.freq_complex = precompute_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)
        
    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"
        
        # (B, seq_len) --> (B, seq_len, dim)
        h = self.tok_embeddings(tokens)
        
        # Retrieve the pairs (m, theta) corresponding to the position [start_pos, start_pos + seq_len]
        freq_complex = self.freq_complex[start_pos:start_pos + seq_len]
        
        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freq_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output