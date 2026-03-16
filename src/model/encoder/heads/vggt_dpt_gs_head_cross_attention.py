# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.encoder.vggt.heads.dpt_head import DPTHead
from .postprocess import postprocess
from einops import rearrange, repeat

class VGGT_DPT_GS_Head_Cross_Attention(DPTHead):
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
                 use_style = False,
                 cond_type = "clip"):
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

        self.use_style = use_style
        self.cond_type = cond_type

        if self.use_style:

            #simpler cross attention on q =content dino patches, kv=style patches
            self.dec_embed_dim = 1024
            self.style_cross_attn = CrocoCrossAttention(
                    token_dim=self.dec_embed_dim,
                    depth=8
                )
            

            # style injector for direct rgb features
            self.style_injector_direct = StylizedFeatureInjector(
                token_dim=self.dec_embed_dim,
                out_channels=128,
                upsample_factor=2
            )

            # style injector for aggregator features
            self.aggregator_embed_dim = 2048
            self.style_injector_aggregator = nn.ModuleList([
                StylizedFeatureInjector(
                token_dim=self.dec_embed_dim,
                out_channels=self.aggregator_embed_dim),
                StylizedFeatureInjector(
                    token_dim=self.dec_embed_dim,
                    out_channels=self.aggregator_embed_dim),
                StylizedFeatureInjector(
                    token_dim=self.dec_embed_dim,
                    out_channels=self.aggregator_embed_dim),
                StylizedFeatureInjector(
                    token_dim=self.dec_embed_dim,
                    out_channels=self.aggregator_embed_dim)
                ])
            
            if self.cond_type=="clip":
                #set clip converter
                self.clip_converter = ClipConverter()


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
            
            # Style vector handling
            style_vec = style_dir[1]
            style_image_shape = style_dir[2]
            clip_vec = style_dir[3] # [1, 512]

            if self.cond_type=="dino":
                style_tokens = style_vec
            elif self.cond_type=="clip":
                style_tokens = self.clip_converter(clip_vec) #[1, n_tok, tok_len]

            _,_,_, style_H, style_W = style_image_shape

            x_tokens = patch_tokens #content tokens; tokens already [B*S, P, tok_len]
            if frames_start_idx is not None and frames_end_idx is not None:
                x_tokens = x_tokens[frames_start_idx:frames_end_idx].contiguous()

            if x_tokens.shape[0] > 1:
                style_tokens = repeat(style_tokens, "b p d -> (b v) p d", v=S)
            patch_style_h, patch_style_w = style_H // self.patch_size[0], style_W // self.patch_size[1]
                        
            stylized_tokens = self.style_cross_attn(x_tokens, style_tokens, patch_h, patch_w, patch_style_h, patch_style_w)  #out (B*S, P, Ctok)
            stylized_feat = stylized_tokens.permute(0, 2, 1).reshape(B * S, -1, patch_h, patch_w) #shape already with B, S

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

            ## HERE inject reshaped, projected styulized tokens
            if self.use_style:
                style_to_inject = self.style_injector_aggregator[dpt_idx](stylized_feat)
                x=x+style_to_inject

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

        if self.use_style:
            # add some stylized features to dorect rgb feat
            stylized_feat_direct = self.style_injector_direct(stylized_feat, target_size = direct_img_feat.shape[-2:])
            direct_img_feat = direct_img_feat + stylized_feat_direct
            
        out = direct_img_feat + out

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        out = self.scratch.output_conv2(out)
        out = out.view(B, S, *out.shape[1:])
        return out

####### CROCO CROSS ATT
from functools import partial
from ..backbone.croco.blocks import DecoderBlock
from src.model.encoder.vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
class CrocoCrossAttention(nn.Module):
    """
    Stacked style cross-attention from croco.
    """
    def __init__(self, token_dim=1024, mlp_ratio=2.0, depth=8, dec_num_heads=8):
        super().__init__()
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        norm_im2_in_dec=True
        # Initialize rotary position embedding if frequency > 0
        rope_freq = 100
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None
        self.dec_blocks = nn.ModuleList([
                DecoderBlock(token_dim, 
                             dec_num_heads, 
                             mlp_ratio=mlp_ratio, 
                             qkv_bias=True, 
                             norm_layer=norm_layer, 
                             norm_mem=norm_im2_in_dec, 
                             rope=self.rope)
                for i in range(depth)])
        # final norm layer
        self.dec_norm = norm_layer(token_dim)
    
    def forward(self, content_feat, style_feat, content_ph, content_pw, style_ph, style_pw):
        
        # content shape (B*S, P, Ctok) 
        # style shape (B*S, Num_style_tokens, Ctok) 
        BS, P, token_size = content_feat.shape
        

        content_pos = None
        if self.rope is not None:
            content_pos = self.position_getter(BS, content_ph, content_pw, device=content_feat.device)
        
        ## RoPo doesnt accept none as pose, I added workaround /home/jk/AnySplat/src/model/encoder/backbone/croco/blocks.py l.160
        style_pos=None
        # if self.rope is not None:
        #     style_pos = self.position_getter(BS, style_ph, style_pw, device=content_feat.device)
        

        for blk in self.dec_blocks:
            content_feat, _ = blk(content_feat,
                                  style_feat,
                                  content_pos,
                                  style_pos)
            
        content_feat = self.dec_norm(content_feat)
        
        return content_feat


### feature injectors

class StylizedFeatureInjector(nn.Module):
    """
    Projects stylized transformer tokens to image space, upsamples them,
    and applies a zero-initialized projection (no fusion here).
    Returns only the processed stylized feature map.
    """
    def __init__(self, token_dim=1024, out_channels=128, upsample_factor=1):
        super().__init__()

        self.upsample_factor = upsample_factor

        # Project transformer tokens to spatial feature map
        self.project = nn.Conv2d(
            in_channels=token_dim,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if self.upsample_factor>1:
            # Upsample + refine stylized features
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=upsample_factor, mode='bilinear', align_corners=False),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            )

        # Zero-initialized projection (the “zero conv”)
        self.zero_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.zero_proj.weight)
        nn.init.zeros_(self.zero_proj.bias)

    def forward(self, stylized_feat, target_size=None):
        """
        stylized_feat: (B*S, token_dim, ph, pw)
        target_size: (H, W)
        Returns: (B*S, out_channels, H, W)
        """
        stylized_feat = self.project(stylized_feat)
        if self.upsample_factor>1:
            stylized_feat = self.upsample(stylized_feat)
        if target_size is not None:
            stylized_feat = F.interpolate(stylized_feat, size=target_size, mode='bilinear', align_corners=True)
        stylized_feat = self.zero_proj(stylized_feat)
        return stylized_feat


## class clip to tokens
class ClipConverter(nn.Module):
    """
     Encoder for injection
    """
    def __init__(self, style_dim=512, hidden_dim=1024, out_channels=1024, n_tokens = 4):
        super().__init__()

        self.out_channels = out_channels
        self.n_tokens = n_tokens
        self.encoder = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_channels*n_tokens)
        )

    def forward(self, style_vec):
        # style_vec: [BS, style_dim]
        feat = self.encoder(style_vec) # [BS, out_channels*n_tokens]
        feat = feat.reshape(-1, self.n_tokens, self.out_channels)
        return feat