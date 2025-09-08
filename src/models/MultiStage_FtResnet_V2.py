from transformers import ResNetModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiStageAdapter(nn.Module):
    """
    Model that takes embeddings from 4 stages of ResNet and converts them into embeddings for Qwen.
    It uses 1x1 convolutions to project each stage to the desired output dimension, upsamples them to the
    same spatial size, concatenates them, and applies a fusion network. Finally, it flattens the spatial dimensions
    and applies a linear interpolation to get the target sequence length. A Layer Normalization is applied at the end.
    """

    def __init__(self, stage_channels=[256, 512, 1024, 2048], out_dim=2048, hidden_multiplier=2, grid_size=14):
        """
        Constructor for MultiStageAdapter.

        Args:
            stage_channels (list): List of channel dimensions for each ResNet stage.
            out_dim (int): Desired output dimension for Qwen embeddings.
            hidden_multiplier (int): Multiplier for the hidden dimension in the fusion network.
        """

        super().__init__()

        # 1x1 convolutions to project each stage to out_dim
        self.projections = nn.ModuleList([
            nn.Conv2d(c, out_dim, kernel_size=1) for c in stage_channels
        ])

        # Fusion network, a small MLP with GELU activation between two linear layers implemented as 1x1 convolutions
        self.fusion = nn.Sequential(
            nn.Conv2d(out_dim * len(stage_channels), out_dim * hidden_multiplier, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_dim * hidden_multiplier, out_dim, kernel_size=1)
        )

        # Adaptive pooling to a fixed grid size before flattening
        self.adaptive_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))

        # Learnable positional embeddings for the flattened grid
        self.pos_embed = nn.Parameter(torch.zeros(1, grid_size * grid_size, out_dim))
        
        # Final Layer Normalization
        self.final_norm = nn.LayerNorm(out_dim)


    def forward(self, stage0, stage1, stage2, stage3, target_seq_len=196):
        """
        Forward pass for MultiStageAdapter.

        Args:
            stage0, stage1, stage2, stage3 (torch.Tensor): Feature maps from the 4 ResNet stages.
            target_seq_len (int): Desired sequence length for the output embeddings.
        """

        # Get spatial dimensions from the last stage
        B, _, Ht, Wt = stage3.shape

        # Project each stage to out_dim and upsample to the size of the last stage
        proj_feats = []
        for feat, proj in zip([stage0, stage1, stage2, stage3], self.projections):
            x = proj(feat)
            x = F.interpolate(x, size=(Ht, Wt), mode='bilinear', align_corners=False)
            proj_feats.append(x)

        # Concatenate along the channel dimension and apply fusion network
        fused = torch.cat(proj_feats, dim=1)  # (B, out_dim*4, Ht, Wt)
        fused = self.fusion(fused)           # (B, out_dim, Ht, Wt)

        # Apply adaptive pooling to get a fixed grid size
        pooled = self.adaptive_pool(fused) # (B, out_dim, grid_size, grid_size)

        # Flatten and permute
        seq = pooled.flatten(2).permute(0, 2, 1) # (B, grid_size*grid_size, out_dim)

        # Add positional embeddings
        seq = seq + self.pos_embed
        
        # Apply final Layer Normalization
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


    def forward(self, pixel_values, target_seq_len=196):
        """
        Forward pass for CompositeModel.
        It extracts features from ResNet and passes them through the MultiStageAdapter.

        Args:
            pixel_values (torch.Tensor): Input image tensor.
            target_seq_len (int): Desired sequence length for the output embeddings.

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
        projected = self.adapter(stage0, stage1, stage2, stage3, target_seq_len)
        
        return projected