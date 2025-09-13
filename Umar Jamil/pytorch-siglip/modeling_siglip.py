from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
    
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: Optional[int] = None ,
        **kwargs
        ):
        
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens
 
class SiglipVisionEmbeddings(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding='valid' # no padding added
        )
        
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        
        self.register_buffer(
            'position_ids',
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False
        )
        
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape # [Batch, Channels, Height, Width]
        
        # [Batch, Channels, Height, Width] -> [Batch, Embed_Dim, Num_Patches_H, Num_Patches_W]
        patch_embeds: torch.Tensor = self.patch_embedding(pixel_values)
        # [Batch, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch, Embed_Dim, Num_Patches]
        # where Num_Patches = Num_Patches_H * Num_Patches_W
        embeddings = patch_embeds.flatten(2)
        # [Batch, Embed_Dim, Num_Patches] -> [Batch, Num_Patches, Embed_Dim]
        embeddings = embeddings.transpose(1, 2)
        # Add position embeddings to each patch
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # [Batch, Embed_Dim, Num_Patches]
        return embeddings
        
class SiglipEncoderLayer(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()        
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # residual: [Batch_size, num_patches, embed_dim]
        residual = hidden_states
        # [Batch_size, num_patches, embed_dim] -> [Batch_size, num_patches, embed_dim]
        hidden_states = self.layer_norm1(hidden_states)
        # hidden_states: [Batch_size, num_patches, embed_dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # [Batch_size, num_patches, embed_dim]
        hidden_states = hidden_states + residual
        # residual: [Batch_size, num_patches, embed_dim]
        residual = hidden_states
        # [Batch_size, num_patches, embed_dim] -> [Batch_size, num_patches, embed_dim]
        hidden_states = self.layer_norm2(hidden_states)
        # [Batch_size, num_patches, embed_dim] -> [Batch_size, num_patches, embed_dim]
        hidden_states = self.mlp(hidden_states)
        # [Batch_size, num_patches, embed_dim] 
        hidden_states = residual + hidden_states
        
        return hidden_states

class SiglipEncoder(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        
    def forward(
        self,
        input_embed: torch.Tensor
    ) -> torch.Tensor:
        # [batch_size, num_patches, embed_dim]
        hidden_states = input_embed
        for layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = layer(hidden_states)
        
        return hidden_states


class SiglipAttention(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # [Batch_size, Num_patches, embed_dim]
        batch_size, seq_len, _ = hidden_states.size()
        # query_states: [Batch_size, Num_patches, embed_dim]
        query_states = self.q_proj(hidden_states)
        # key_states: [Batch_size, Num_patches, embed_dim]
        key_states = self.k_proj(hidden_states)
        # value_states: [Batch_size, Num_patches, embed_dim]
        value_states = self.v_proj(hidden_states)
        # query_states: : [Batch_size, Num_heads, Num_patches, Head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Calculate the attention using the formula Q * K^T / sqrt(d_k). attention_weights: [batch_size, num_heads, num_patches, num_patches]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f"{attn_weights.size()}"
            )
        
        # Apply the softmax row-wise. attention_weights: [batch_size, num_heads, num_patches, num_patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply the dropout only during training
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # Multiply the attention weights by the value states. attn_output: [batch_size, num_heads, num_patches, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)
        
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"attn_output should be of size {batch_size, self.num_heads, seq_len, self.head_dim}, but is"
                f"{attn_output.size()}"
            )
        
        # [batch_size, num_heads, num_patches, head_dim] -> [batch_size, num_patches, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [batch_size, num_patches, num_heads, head_dim] -> [batch_size, num_patches, embed_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        # [batch_size, num_patches, embed_dim]
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights
        


class SiglipMLP(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_size, num_patches, embed_dim] -> [Batch_size, num_patches, intermediated_size]
        hidden_states = self.fc1(hidden_states)
        # hidden_states: [Batch_size, num_patches, intermediated_size]
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # [Batch_size, num_patches, intermediated_size] -> [Batch_size, num_patches, embed_dim]
        hidden_states = self.fc2(hidden_states)
        
        return hidden_states
        
        
class SiglipVisionTransformer(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig,):
        super().__init__()
        self.config = config
        
        emb_dim = config.hidden_size
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoer(config)
        self.post_layernorm = nn.LayerNorm(emb_dim, eps=config.layer_norm_eps)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        
        # pixel_values: [Batch, Channels, Height, Width] -> [Batch, Num_Patches, Embed_dim]
        hidden_states = self.embeddings(pixel_values)
        
        last_hidden_states = self.encoder(hidden_states)
        
        last_hidden_states = self.post_layernorm(last_hidden_states)
        
        return last_hidden_states


class SiglipVisionModel(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
        
    def forward(self, pixel_values) -> Tuple:
        
        # [Batch, Channels, Height, Width] -> [Batch, Num_Patches, Embed_dim]
        return self.vision_model(pixel_values=pixel_values)