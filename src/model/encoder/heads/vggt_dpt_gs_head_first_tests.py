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

        self.use_style = use_style
        if self.use_style:

            # # # injects token features using zero conv
            self.zero_conv_proj = nn.Sequential(
                nn.Conv2d(128, 128, 1),
                # nn.InstanceNorm2d(128, affine=True), ##these lines were used in working impl, but i think it should be skipped?? ESpecially this norm
                nn.LeakyReLU(0.1, inplace=True) ##these lines were used in working impl, but i think it should be skipped??
            )
            nn.init.zeros_(self.zero_conv_proj[0].weight)
            nn.init.zeros_(self.zero_conv_proj[0].bias)

            #simpler cross attention on q =content dino patches, kv=style patches
            self.dec_embed_dim = 1024
            self.style_cross_attn = CrocoCrossAttention(
                    token_dim=self.dec_embed_dim,
                    depth=8
                )
            
            #projects stylized tokens to image space
            self.style_project = nn.Conv2d(
                in_channels=self.dec_embed_dim,  # e.g. 1024 from transformer tokens
                out_channels=128,    # same as first fusion channel (e.g. 256)
                kernel_size=1,
                stride=1,
                padding=0,
            )

            self.upsample_stylized_features = nn.Sequential(
                ##transpose do NOT converge, even with MSE
                # nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=1),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(128, 128, kernel_size=3, padding=1),
                # nn.ReLU(inplace=True),
                # nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4, padding=1),
                # nn.ReLU(inplace=True),
                # nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=1),
                # nn.ReLU(inplace=True)

                ## this used and kind of work
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(128, 128, 3, padding=1),
                # nn.ReLU(inplace=True),
                # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                # nn.Conv2d(128, 128, 3, padding=1),
                # nn.ReLU(inplace=True),
                # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                # nn.Conv2d(128, 128, 3, padding=1),

                ##experimental with transpose, not converge
                # # Stage 1 – smooth base (bilinear upsample)
                # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                # nn.Conv2d(128, 128, kernel_size=3, padding=1),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(128, 128, kernel_size=3, padding=1),
                # nn.ReLU(inplace=True),

                # # Stage 2 – another bilinear + conv, adds mid-level structure
                # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                # nn.Conv2d(128, 128, kernel_size=3, padding=1),
                # nn.ReLU(inplace=True),

                # # Stage 3 – learned transposed conv for fine detail and texture
                # nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(128, 128, kernel_size=3, padding=1),

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

        #direct RGB 
        direct_img_feat = self.input_merger(imgs.flatten(0, 1))

        ## apply stylization
        if style_dir is not None and self.use_style:
            
            # Style vector handling
            style_vec = style_dir[1]
            style_image_shape = style_dir[2]
            _,_,_, style_H, style_W = style_image_shape

            ## some handling of clip 512 vector
            # if style_vec.dim() == 1:
            #     style_vec = style_vec.unsqueeze(0)
            # if style_vec.size(0) == 1 and (B * S) > 1:
            #     style_vec = style_vec.expand(B * S, -1)

            
            x_tokens = patch_tokens #patch tokens already B*S, P, tok_len
            if frames_start_idx is not None and frames_end_idx is not None:
                x_tokens = x_tokens[frames_start_idx:frames_end_idx].contiguous()

            #style tokens
            # style_vec = patch_tokens[:1].contiguous() #firs frame from scene as style
            
            if x_tokens.shape[0] > 1:
                style_vec = style_vec.expand(x_tokens.shape[0], -1, -1)
            patch_style_h, patch_style_w = style_H // self.patch_size[0], style_W // self.patch_size[1]
            
            # style_vec = x_tokens #all frames from scene as style
            
            stylized_tokens = self.style_cross_attn(x_tokens, style_vec, patch_h, patch_w, patch_style_h, patch_style_w)  #out (B*S, P, Ctok)

            stylized_feat = stylized_tokens.permute(0, 2, 1).reshape(B * S, -1, patch_h, patch_w) #shape already with B, S
            stylized_feat = self.style_project(stylized_feat)
            stylized_feat = self.upsample_stylized_features(stylized_feat)
            stylized_feat =F.interpolate(stylized_feat, size=(H, W), mode='bilinear', align_corners=True)

            ### different options to make final features

            # original + inject using zero-conv
            out = direct_img_feat + out + self.zero_conv_proj(stylized_feat)

            # # only stylized  
            # out = stylized_feat

            # stylized + direct
            # out = stylized_feat + direct_img_feat

            # intermediate + stylized
            # out = stylized_feat + out
            
            # skip out features from intermediate layers, nut use zero-conv injection
            # out = direct_img_feat + self.zero_conv_proj(stylized_feat)

            #original
            # out = direct_img_feat + out

            #only aggregator
            # out = out

            #only direct
            # out = direct_img_feat

        else:
            # original features
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

#########################

#thrashy mock implementation below

class MockAttention(nn.Module):
    """
    One layer of style-conditioned cross-attention + MLP with residuals.
    No its thrash only, but we keep it for the style projection -its coded and ready
    """
    def __init__(self, style_dim=128, token_dim=1024, mlp_ratio=2.0):
        super().__init__()
        hidden_dim = 512
        self.style_dim=style_dim
        self.style_token_dim =token_dim #apparently token dim for content and style need to match... :/
        self.style_proj = nn.Sequential(
            nn.Linear(style_dim, hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.style_token_dim),
        )

        self.mlp = nn.Sequential(
            nn.Linear(token_dim, int(token_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(token_dim * mlp_ratio), token_dim),
        )
        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)

       

    def forward(self, tokens, style):
        """
        tokens: (B*S, P, C_tok)
        style:  (B, style_dim)
        """
        B_times_S, P, C_tok = tokens.shape
        return tokens


        # Project style vector
        style_proj = self.style_proj(style[:,:self.style_dim])  # (B*S, token_size)
        style_proj = self.norm1(style_proj)  # (B*S, token_size)
        style_proj = style_proj.unsqueeze(1).repeat(1,P,1) #(B*S, P, token_size)
  

        # Cross-attention: tokens query style
        tokens = tokens + style_proj

        # # Feed-forward (MLP)
        # mlp_out = self.mlp(tokens)
        # tokens = tokens + self.norm2(mlp_out)

        return tokens


