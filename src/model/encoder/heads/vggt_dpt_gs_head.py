# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.encoder.vggt.heads.dpt_head import DPTHead
from .postprocess import postprocess


class VGGT_DPT_GS_Head(DPTHead):
    def __init__(self,
                 dim_in: int,
                 patch_size: int = 14,
                 output_dim: int = 83,
                 activation: str = "inv_log",
                 conf_activation: str = "expp1",
                 features: int = 256,
                 out_channels: List[int] = [256, 512, 1024, 1024],
                 intermediate_layer_idx: List[int] = [4, 11, 17, 23],
                 pos_embed: bool = True,
                 feature_only: bool = False,
                 down_ratio: int = 1,
                 use_style = False):
        super().__init__(dim_in, patch_size, output_dim, activation, conf_activation,
                         features, out_channels, intermediate_layer_idx, pos_embed, feature_only, down_ratio)

        head_features_1 = 128
        head_features_2 = 128 if output_dim > 50 else 32

        self.input_merger = nn.Sequential(
            nn.Conv2d(3, head_features_2, 7, 1, 3),
            nn.ReLU(),
        )

        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
        )


    def forward(self, encoder_tokens: List[torch.Tensor], depths, imgs, patch_start_idx: int = 5,
                image_size=None, conf=None, frames_chunk_size: int = 8, style_dir=None, patch_tokens=None):

        B, S, _, H, W = imgs.shape
        image_size = self.image_size if image_size is None else image_size

        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(encoder_tokens, imgs, patch_start_idx, style_dir=style_dir, patch_tokens=patch_tokens)

        all_preds = []
        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)
            chunk_output = self._forward_impl(
                encoder_tokens, imgs, patch_start_idx,
                frames_start_idx, frames_end_idx, style_dir=style_dir, patch_tokens=patch_tokens
            )
            all_preds.append(chunk_output)
        return torch.cat(all_preds, dim=1)

    def _forward_impl(self, encoder_tokens: List[torch.Tensor], imgs,
                      patch_start_idx: int = 5,
                      frames_start_idx: int = None,
                      frames_end_idx: int = None,
                      style_dir=None,
                      patch_tokens=None):

        if frames_start_idx is not None and frames_end_idx is not None:
            imgs = imgs[:, frames_start_idx:frames_end_idx]

        B, S, _, H, W = imgs.shape
        patch_h, patch_w = H // self.patch_size[0], W // self.patch_size[1]

         ## apply stylization of tokens
        if style_dir is not None and self.use_style:
            clip_vec = style_dir[3] # [1, 512]

        # === DPT multi-layer fusion path
        out = []
        dpt_idx = 0
        for layer_idx in self.intermediate_layer_idx:
            if len(encoder_tokens) > 10:
                x_base = encoder_tokens[layer_idx][:, :, patch_start_idx:]
            else:
                list_idx = self.intermediate_layer_idx.index(layer_idx)
                x_base = encoder_tokens[list_idx][:, :, patch_start_idx:]

            if frames_start_idx is not None and frames_end_idx is not None:
                x_base = x_base[:, frames_start_idx:frames_end_idx].contiguous()
            
            x_base = x_base.view(B * S, -1, x_base.shape[-1]) # (B*S, P, Ctok)
            x_base = self.norm(x_base)

            # copy tokens
            x = x_base
            
            # default behavior
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)) # (B*S, Ctok, ph, pw)

            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)    

            out.append(x)
            dpt_idx += 1

        # Fuse features and upsample
        out = self.scratch_forward(out)  # (B*S, 128, h’, w’)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)

        direct_img_feat = self.input_merger(imgs.flatten(0, 1))

        out = direct_img_feat + out

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        out = self.scratch.output_conv2(out)
        out = out.view(B, S, *out.shape[1:])
        return out