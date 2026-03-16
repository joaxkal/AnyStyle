import torch
import torch.nn as nn
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor
from torchvision import transforms
from dataclasses import dataclass
from src.utils.vgg_model import VGGEncoder, calc_mean_std

from src.dataset.types import BatchedExample
from src.model.types import Gaussians
from src.loss import Loss
from src.model.decoder.decoder import DecoderOutput
from src.misc.nn_module_tools import convert_to_buffer

from PIL import Image

@dataclass
class LossStyleCfg:
    style_weight: float
    content_weight: float

@dataclass
class LossStyleCfgWrapper:
    style: LossStyleCfg

class LossStyle(Loss[LossStyleCfg, LossStyleCfgWrapper]):
    
    def __init__(self, cfg: LossStyleCfgWrapper) -> None:
        super().__init__(cfg)
        
        self.vgg = VGGEncoder()
        self.vgg.to("cuda")
        convert_to_buffer(self.vgg, persistent=False)
        
        self.preprocess2 = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

        #add resize to 224 -vgg was trained on that. TESTING
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    
    def forward(
        self,
        style_img,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict | None,
        global_step: int,
        mask_for_style_loss: Tensor | None
    ) -> tuple[Float[Tensor, ""], Float[Tensor, ""]]:
        
        b, v, _, _, _ = batch["context"]["image"].shape
        
        target_img = (batch["context"]["image"][:, batch["using_index"]] + 1) / 2
        target_img = rearrange(target_img, "b v c h w -> (b v) c h w")
        pred_img = rearrange(prediction.color, "b v c h w -> (b v) c h w")

        # [0, 1] -> ImageNet mean/std
        target_img = self.preprocess(target_img)
        pred_img = self.preprocess(pred_img)
        
        # extract vgg feature maps for all
        pred_img_features = self.vgg(pred_img)
        target_img_features = self.vgg(target_img)
        
        # use second last layer for content loss, which yields the best results on a single scene for now
        content_loss = 0
        content_loss += nn.functional.mse_loss(pred_img_features[-2], target_img_features[-2])
        content_loss += nn.functional.mse_loss(pred_img_features[-1], target_img_features[-1])
        
        # calculate style loss with pred_img and style_img
        style_loss = torch.zeros((), device=target_img.device)
        print("********", self.cfg.style_weight)

        if self.cfg.style_weight > 0:

            if mask_for_style_loss is not None:
                style_mask = mask_for_style_loss.float().view(b, 1)   # (B, 1)
                style_mask = repeat(style_mask, 'b 1 -> (b v)', v=v)  # (B*V,)
                style_mask = style_mask.to(target_img.device)
            else:
                style_mask = torch.ones(
                    b * v,
                    device=target_img.device,
                    dtype=torch.float32
                )

            style_img = self.preprocess(style_img)
            style_img_features = self.vgg(style_img)
            for pred_img_feature, style_img_feature in zip(pred_img_features, style_img_features):
                pred_img_feature_mean, pred_img_feature_std = calc_mean_std(pred_img_feature)
                style_img_feature_mean, style_img_feature_std = calc_mean_std(style_img_feature)
                
                #repeat style features here, instead of forwarding multiple times the same style image, prevents oom
                style_img_feature_mean = repeat(style_img_feature_mean, 'b d1 d2 -> (b v) d1 d2', v=v)
                style_img_feature_std = repeat(style_img_feature_std, 'b d1 d2 -> (b v) d1 d2', v=v)
                
                # per-sample losses
                mean_loss = (pred_img_feature_mean - style_img_feature_mean).pow(2).mean(dim=(1,2))
                std_loss  = (pred_img_feature_std  - style_img_feature_std ).pow(2).mean(dim=(1,2))

                mean_loss = (mean_loss * style_mask).sum()
                std_loss  = (std_loss  * style_mask).sum()

                denom = style_mask.sum().clamp_min(1.0)
                style_loss += (mean_loss + std_loss) / denom
        
        return self.cfg.style_weight * style_loss, self.cfg.content_weight * content_loss