# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict, Any

from src.model.encoder.vggt.layers.block import Block
from src.model.encoder.vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from einops import rearrange, repeat

logger = logging.getLogger(__name__)


class Aggregator_injection(nn.Module):
    """
    TODO!!!!!! INHERIT
    """

    def __init__(
        self,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
    ):
        super().__init__()
        self.use_checkpoint = True

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None
        
        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # style injector for aggregator features
        self.aggregator_embed_dim = 1024
        self.injection_layers_idx=[0,4,11,17,23]
        self.style_injector_aggregator = nn.ModuleList([
            StyleInjector(
                out_channels=self.aggregator_embed_dim, hidden_dim =1024) for i in self.injection_layers_idx
            ])
        
    

    def forward(
        self,
        ready_package,
        style_dir,
        intermediate_layer_idx: Optional[List[int]] = None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            ready to use package with everything 

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        tokens, B, S, P, C, pos, patch_start_idx = ready_package

        frame_idx = 0
        global_idx = 0
        output_list = []
        layer_idx = 0
        
        # Convert intermediate_layer_idx to a set for O(1) lookup
        if intermediate_layer_idx is not None:
            required_layers = set(intermediate_layer_idx)
            # Always include the last layer for camera_head
            required_layers.add(self.depth - 1)

        for _ in range(self.aa_block_num):

            #clip injection
            clip_vec = style_dir[3]
            if layer_idx in self.injection_layers_idx:
                injection_idx_mapped = self.injection_layers_idx.index(layer_idx)
                
                tokens = tokens.view(B, S, P, C)

                #inject style only to patch tokens, not to first camera / register tokens (patch_start_idx)
                style_to_inject = self.style_injector_aggregator[injection_idx_mapped](clip_vec, target_size = [B, S, (P-patch_start_idx)])    
                
                tokens[:, :, patch_start_idx:,] += style_to_inject
                tokens = tokens.view(B, S*P, C)

            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            if intermediate_layer_idx is not None:
                for i in range(len(frame_intermediates)):
                    current_layer = layer_idx + i
                    if current_layer in required_layers:
                        # concat frame and global intermediates, [B x S x P x 2C]
                        concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                        output_list.append(concat_inter)
                layer_idx += self.aa_block_size
            
            else:
                for i in range(len(frame_intermediates)):
                    # concat frame and global intermediates, [B x S x P x 2C]
                    concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                    output_list.append(concat_inter)
        
        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []
        
        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.use_checkpoint:
                tokens = torch.utils.checkpoint.checkpoint(
                    self.frame_blocks[frame_idx],
                    tokens,
                    pos,
                    use_reentrant=False,
                )
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []
        
        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.use_checkpoint:
                tokens = torch.utils.checkpoint.checkpoint(
                    self.global_blocks[global_idx],
                    tokens,
                    pos,
                    use_reentrant=False,
                )
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates
    


class StyleInjector(nn.Module):
    """
    """
    def __init__(self, out_channels=128, hidden_dim=256):
        super().__init__()
        # self.clip_dim = 768  # for ViT-L/14
        self.clip_dim = 512  # for ViT-B/32
        self.hidden_dim = hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(self.clip_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, out_channels)
        )

        # Zero-initialized projection (the “zero conv”)
        self.zero_proj = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.zero_proj.weight)
        nn.init.zeros_(self.zero_proj.bias)

    def forward(self, clip_vec, target_size=None):
        """
        clip vec: (B, clip_dim)
        target_size: (B,S, P)
        Returns: B, S,token_dim,P
        """
        B,S,P = target_size
        projected_clip = self.fc(clip_vec)
        projected_clip = repeat(projected_clip, "b d -> (b v) d", v=S)
        projected_clip = projected_clip.unsqueeze(-1).expand(B*S, -1, P)
        projected_clip = self.zero_proj(projected_clip) # [B*S, C, P]
        return projected_clip.permute(0,2,1).view(B, S, P, -1)  #B, S, P, C
    