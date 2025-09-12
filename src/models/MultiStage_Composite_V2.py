from transformers import ResNetModel
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn as nn
import torch.nn.functional as F


class MultiStageAdapter(nn.Module):
    """
    Model that takes embeddings from four stages of ResNet and converts them into embeddings suitable for Qwen.
    It uses 1x1 convolutions to project each stage to a common dimension, upsamples them to the size of the last stage,
    concatenates them, and applies a fusion network followed by adaptive pooling and positional embeddings.
    The output is a sequence of embeddings ready for Qwen.

    This version incorporates a Transformer Encoder for self-attention to capture global relationships
    in addition to the convolutional fusion network for local context.
    """

    def __init__(self, stage_channels=[256, 512, 1024, 2048], out_dim=2048, hidden_multiplier=2, grid_size=14, n_heads=8, num_attention_layers=2):
        """
        Constructor for the AdvancedMultiStageAdapter.

        Args:
            stage_channels (list): Channel dimensions for each ResNet stage.
            out_dim (int): The desired output feature dimension.
            hidden_multiplier (int): Multiplier for the fusion network's hidden layer.
            grid_size (int): The side length of the output grid (e.g., 14 for a 14x14 grid -> 196 tokens).
            n_heads (int): Number of attention heads in the Transformer Encoder. Must be a divisor of out_dim.
            num_attention_layers (int): Number of layers in the Transformer Encoder.
        """
        
        super().__init__()

        # Projections to unify channel dimensions
        self.projections = nn.ModuleList([
            nn.Conv2d(c, out_dim, kernel_size=1) for c in stage_channels
        ])

        # Fusion network with 1x1 convolution for local context
        self.fusion = nn.Sequential(
            nn.Conv2d(out_dim * len(stage_channels), out_dim * hidden_multiplier, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_dim * hidden_multiplier, out_dim, kernel_size=1)
        )

        # Transformer Encoder for global context (self-attention)
        encoder_layer = TransformerEncoderLayer(d_model=out_dim, nhead=n_heads, batch_first=True)
        self.attention = TransformerEncoder(encoder_layer, num_layers=num_attention_layers)

        # Adaptive pooling to create a fixed-size grid
        self.adaptive_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))

        # Learnable positional embeddings for the grid tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, grid_size * grid_size, out_dim))

        # Final layer normalization
        self.final_norm = nn.LayerNorm(out_dim)


    def forward(self, stage0, stage1, stage2, stage3):
        """
        Forward pass for the AdvancedMultiStageAdapter.

        Args:
            stage0, stage1, stage2, stage3 (torch.Tensor): Feature maps from the 4 ResNet stages.

        Returns:
            torch.Tensor: Output embeddings suitable for Qwen.
        """

        B, _, Ht, Wt = stage3.shape

        # Project and upsample features from each stage
        proj_feats = []
        for feat, proj in zip([stage0, stage1, stage2, stage3], self.projections):
            x = proj(feat)
            x = F.interpolate(x, size=(Ht, Wt), mode='bilinear', align_corners=False)
            proj_feats.append(x)

        # Concatenate and apply the convolutional fusion network
        fused = torch.cat(proj_feats, dim=1)
        fused = self.fusion(fused)  # Shape: (B, C, H, W)

        # Apply self-attention to learn global relationships
        B, C, H, W = fused.shape
        # Reshape for attention: (B, C, H, W) -> (B, L, C) where L = H*W
        fused_seq = fused.flatten(2).permute(0, 2, 1)
        attended_seq = self.attention(fused_seq)
        # Reshape back to image format: (B, L, C) -> (B, C, H, W)
        attended_img = attended_seq.permute(0, 2, 1).view(B, C, H, W)
        
        # Apply adaptive pooling to get a fixed grid size
        pooled = self.adaptive_pool(attended_img) # Shape: (B, C, grid_size, grid_size)

        # Flatten grid to sequence and add positional embeddings
        seq = pooled.flatten(2).permute(0, 2, 1) # Shape: (B, grid_size*grid_size, C)
        seq = seq + self.pos_embed
        
        # Apply final layer normalization
        seq = self.final_norm(seq)
        
        return seq


class CompositeModel(nn.Module):
    """
    Composite model that integrates ResNet for image feature extraction and MultiStageAdapter
    to convert these features into embeddings suitable for Qwen.
    """

    def __init__(self):
        """
        Constructor for CompositeModel.
        Initializes the ResNet model and the MultiStageAdapter.
        """

        super().__init__()
        self.resnet = ResNetModel.from_pretrained("microsoft/resnet-50")
        self.adapter = MultiStageAdapter()


    def forward(self, pixel_values):
        """
        Forward pass for CompositeModel.
        It extracts features from ResNet and passes them through the MultiStageAdapter.

        Args:
            pixel_values (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Output embeddings suitable for Qwen.
        """

        intermediate_outputs = {}

        # Register hooks to capture outputs from each ResNet stage
        def get_hook(idx):
            def hook(module, input, output):
                intermediate_outputs[f"stage_{idx}"] = output
            return hook

        # Register hooks for each stage
        hooks = []
        for idx, stage in enumerate(self.resnet.encoder.stages):
            hooks.append(stage.register_forward_hook(get_hook(idx)))

        # Forward pass through ResNet to get intermediate features
        intermediate_outputs.clear()
        _ = self.resnet(pixel_values)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Extract features from each stage
        stage0, stage1, stage2, stage3 = (
            intermediate_outputs["stage_0"],
            intermediate_outputs["stage_1"],
            intermediate_outputs["stage_2"],
            intermediate_outputs["stage_3"],
        )

        # Pass the features through the MultiStageAdapter
        projected = self.adapter(stage0, stage1, stage2, stage3)
        
        return projected