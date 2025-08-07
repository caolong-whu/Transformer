import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # vocab_size:词汇表的长度
        # d_model:每个词的向量维度，512
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    
    # d_model: 每个词的向量维度，512
    # seq_len: 输入sentence的最大单词数量
    # dropout: 防止过拟合
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Creat a matrix of shape (seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        # Creat a vector of shape (seq_len) (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=float).unsqueeze(1)
        # Create the div term of positional encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000)/d_model))
        # Apply the sin to even positions
        pe[0, 0::2] = torch.sin(position * div_term)
        pe[0, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        
        # 将pe保存到模型的state_dict()中，但是不参与更新
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.alph = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alph * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(self.linear_1(x)))
    
class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not devided by h !"
        
        self.d_k = d_model / h
        self.w_q = nn.Linear(d_model, d_model) #Wq
        self.w_k = nn.Linear(d_model, d_model) #Wk
        self.w_v = nn.Linear(d_model, d_model) #Wv
        
        self.w_o = nn.Linear(d_model, d_model) #Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: float):
        
        d_k = query.shape[-1]
        
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores: torch.Tensor = query @ key.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, 1e-9)
        attention_scores = attention_scores.softmax(dim=-1) # (Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores
        
        
        
    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key = self.w_k(k)   # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        value = self.w_v(v) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        
        # (Batch, seq_len, d_model) --> (Batch, seq_len, h, d_k) --> (Batch, h, seq_len, d_k) 每个头能够看到整个句子的d_k个信息
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        x, attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k) # contiguous将张量重新复制一份连续的内存，因为view必须要内存连续才行。transpose只是将维度表示进行了改变，但内存布局没有改
        # 其他写法
        # x = x.transpose(1, 2).reshape(x.shape[0], -1, self.h * self.d_k)
        # x = x.permute(0, 2, 1, 3).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        # x = x.permute(0, 2, 1, 3).flatten(2, 3)
        
        
        
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)