"""
OmniCoreX Core Model Definition

This module defines the core neural architecture of OmniCoreX - the ultimate AI brain
integrating infinite knowledge streams with unparalleled adaptive reasoning and
real-time decision making.

Features:
- Multi-stream knowledge integration layers for combining diverse inputs.
- Adaptive reasoning modules enabling dynamic context-aware inference.
- Hierarchical transformer blocks optimized for scalability and modularity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiStreamEncoder(nn.Module):
    """
    Encodes multiple knowledge streams with dedicated encoders and fuses representations.
    """

    def __init__(self, stream_configs, embed_dim):
        """
        Args:
            stream_configs (dict): {stream_name: input_dim} mapping input sizes per stream.
            embed_dim (int): Dimension of embedding vectors after encoding.
        """
        super().__init__()
        self.stream_encoders = nn.ModuleDict()
        for name, input_dim in stream_configs.items():
            self.stream_encoders[name] = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(inplace=True)
            )
        self.fusion_layer = nn.Linear(len(stream_configs)*embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, input_streams):
        """
        Args:
            input_streams (dict): {stream_name: tensor of shape (batch_size, seq_len, input_dim)}

        Returns:
            fused_embed: Tensor of shape (batch_size, seq_len, embed_dim)
        """
        encoded_streams = []
        for name, encoder in self.stream_encoders.items():
            x = input_streams[name]  # shape: (batch, seq_len, input_dim)
            encoded = encoder(x)      # (batch, seq_len, embed_dim)
            encoded_streams.append(encoded)
        concat_embeds = torch.cat(encoded_streams, dim=-1)  # (batch, seq_len, embed_dim * n_streams)
        fused = self.fusion_layer(concat_embeds)            # (batch, seq_len, embed_dim)
        fused = self.norm(fused)
        return fused


class AdaptiveReasoningBlock(nn.Module):
    """
    Transformer block with adaptive reasoning capability through dynamic gating
    and context-modulated feed-forward networks.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Context gating mechanism dynamically modulates FFN
        self.context_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, context=None, attn_mask=None, key_padding_mask=None):
        """
        Args:
            x: Input tensor (seq_len, batch_size, embed_dim)
            context: Optional context tensor (seq_len, batch_size, embed_dim) for gating.
            attn_mask: Optional attention mask.
            key_padding_mask: Optional padding mask.

        Returns:
            Tensor of shape (seq_len, batch_size, embed_dim)
        """
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = self.norm1(x)

        # Context gating
        gate = self.context_gate(context) if context is not None else torch.ones_like(x)
        ffn_out = self.ffn(x)
        x = x + gate * ffn_out
        x = self.norm2(x)
        return x


class OmniCoreXModel(nn.Module):
    """
    Core OmniCoreX Model combining multi-stream encoding and adaptive reasoning layers.

    Arguments:
        stream_configs (dict): Dictionary of input stream names and their feature dims.
        embed_dim (int): Embedding dimension for all transformer layers.
        num_layers (int): Number of adaptive reasoning transformer blocks.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
    """

    def __init__(self,
                 stream_configs,
                 embed_dim=768,
                 num_layers=24,
                 num_heads=12,
                 dropout=0.1):
        super().__init__()

        # Multi-stream encoder to fuse heterogeneous knowledge sources
        self.encoder = MultiStreamEncoder(stream_configs, embed_dim)

        # Stack of adaptive reasoning transformer layers
        self.layers = nn.ModuleList([
            AdaptiveReasoningBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Output projection to vocabulary or downstream embedding space
        self.output_head = nn.Linear(embed_dim, embed_dim)  # Placeholder, plug model head as needed

    def forward(self, input_streams, context_streams=None, attn_mask=None, key_padding_mask=None):
        """
        Forward pass through OmniCoreX.

        Args:
            input_streams (dict): Input tensors keyed by stream name.
            context_streams (dict or None): Optional context passed to reasoning blocks.
            attn_mask (Tensor or None): Optional attention mask.
            key_padding_mask (Tensor or None): Optional key padding mask.

        Returns:
            output: Tensor shaped (batch_size, seq_len, embed_dim)
        """
        # Encode multi-stream inputs
        x = self.encoder(input_streams)  # (batch, seq_len, embed_dim)
        # Change to seq_len, batch, embed_dim for transformer layers
        x = x.transpose(0, 1)  # (seq_len, batch, embed_dim)

        # Prepare context tensors if given for each layer or None
        if context_streams is not None:
            context_embeds = self.encoder(context_streams).transpose(0, 1)
        else:
            context_embeds = None

        for layer in self.layers:
            x = layer(x, context=context_embeds, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        x = self.norm(x)
        x = x.transpose(0, 1)  # back to (batch, seq_len, embed_dim)
        output = self.output_head(x)
        return output


if __name__ == "__main__":
    # Simple test run with dummy data
    batch_size = 2
    seq_len = 16
    stream_configs = {
        "text": 128,
        "image": 256,
        "sensor": 64
    }
    model = OmniCoreXModel(stream_configs=stream_configs, embed_dim=128, num_layers=4, num_heads=4)

    # Generate dummy inputs per stream
    inputs = {
        name: torch.randn(batch_size, seq_len, input_dim)
        for name, input_dim in stream_configs.items()
    }

    outputs = model(inputs)
    print(f"Output shape: {outputs.shape}")  # Expected (batch_size, seq_len, embed_dim)

