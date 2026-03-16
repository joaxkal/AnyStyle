from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor
from torchvision import transforms

from src.dataset.types import BatchedExample
from src.misc.nn_module_tools import convert_to_buffer
from src.model.decoder.decoder import DecoderOutput
from src.model.types import Gaussians
from src.utils.clip_utils import get_style_embedding, load_model
from src.utils.clip_utils.image_utils import clip_normalize, img_normalize
from src.utils.clip_utils.vgg import get_features

from .loss import Loss
from einops import rearrange, repeat

@dataclass
class DirCLIPLossCfg:
    weight: float
    num_crop_per_img: int = 4
    patch_loss_mult: float = 1.0
    crop_size: int = 128


@dataclass
class DirCLIPLossCfgWrapper:
    clip_dir: DirCLIPLossCfg


class DirCLIPLoss(Loss[DirCLIPLossCfg, DirCLIPLossCfgWrapper]):
    def __init__(self, cfg: DirCLIPLossCfgWrapper) -> None:
        super().__init__(cfg)

        self.num_crops = self.cfg.num_crop_per_img  # per image in scene. If scene has 16 views we have N*16 patches
        self.patch_loss_mult = self.cfg.patch_loss_mult
        self.crop_size = self.cfg.crop_size

        self.cropper = transforms.Compose([transforms.RandomCrop(self.crop_size)])
        self.augment = transforms.Compose(
            [transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5), transforms.Resize(224)]
        )

    def forward(
        self,
        clip_model,
        target_clip_dir,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict | None,
        global_step: int,
        adapter = None
    ) -> Float[Tensor, ""]:
        
        # Rearrange predicted and ground truth images
        B, V, C, H, W = prediction.color.shape
        pred_img = rearrange(prediction.color, "b v c h w -> (b v) c h w")
        
        gt_img = (batch["context"]["image"][:, batch["using_index"]] + 1) / 2
        gt_img = rearrange(gt_img, "b v c h w -> (b v) c h w")

        # rearrange clip vec to (bv, dim)
        target_clip_dir = repeat(target_clip_dir, 'b d -> (b v) d', v=V)

        ### GLOBAL loss
        pred_img_features = clip_model.encode_image(clip_normalize(pred_img)) #[BV, D]
        if adapter is not None:
            # we have normally disabled autocast for all losses, so we need to autocast this place
            with torch.amp.autocast("cuda", enabled=True):
                pred_img_features = adapter(pred_img_features, modality = "image")
        pred_img_features /= pred_img_features.clone().norm(dim=-1, keepdim=True)

        gt_img_features = clip_model.encode_image(clip_normalize(gt_img))
        if adapter is not None:
            # we have normally disabled autocast for all losses, so we need to autocast this place
            with torch.amp.autocast("cuda", enabled=True):    
                gt_img_features = adapter(gt_img_features, modality = "image")
        gt_img_features /= gt_img_features.clone().norm(dim=-1, keepdim=True)

        features_dir = pred_img_features - gt_img_features
        features_dir /= features_dir.clone().norm(dim=-1, keepdim=True)

        loss = (1.0 - torch.cosine_similarity(features_dir, target_clip_dir)).mean()

        if self.num_crops == 0:
            return self.cfg.weight * (torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0))

        ### PATCHES

        B = pred_img.shape[0]
        pred_img_expanded = repeat(pred_img, 'bv c h w -> (bv n) c h w', n=self.num_crops)
        img_patch = self.augment(self.cropper(pred_img_expanded))

        patch_image_features = clip_model.encode_image(clip_normalize(img_patch)) #[BVN, D]
        if adapter is not None:
            # we have normally disabled autocast for all losses, so we need to autocast this place
            with torch.amp.autocast("cuda", enabled=True):    
                patch_image_features = adapter(patch_image_features, modality = "image")
        patch_image_features /= patch_image_features.clone().norm(dim=-1, keepdim=True)

        gt_img_features_expanded = repeat(gt_img_features, 'bv d -> (bv n) d', n=self.num_crops)
        patch_img_direction = patch_image_features - gt_img_features_expanded
        patch_img_direction /= patch_img_direction.clone().norm(dim=-1, keepdim=True)

        target_clip_dir_expanded = repeat(target_clip_dir, 'bv d -> (bv n) d', n=self.num_crops)

        loss_temp = 1.0 - torch.cosine_similarity(
            patch_img_direction, target_clip_dir_expanded, dim=1
        )
        loss_patch = loss_temp.mean()

        return self.cfg.weight * (
            torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
            + torch.nan_to_num(loss_patch, nan=0.0, posinf=0.0, neginf=0.0)*self.patch_loss_mult
        )
