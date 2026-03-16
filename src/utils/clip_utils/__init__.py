import torch
from src.longclip.model import longclip
from PIL import Image

from src.utils.clip_utils.template import (
    imagenet_templates,
    imagenet_templates_small,
)
from src.utils.clip_utils.image_utils import load_image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
    


TRUNCATE_BOOL = True  # kept for compatibility, not used by LongCLIP
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

n_px = 224
_transform = Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


# ============================================================
# Model Loading
# ============================================================

def load_model(
    ckpt_path="src/longclip/checkpoints/longclip-B-32.pt",
    device=DEVICE,
):
    model, _ = longclip.load(ckpt_path, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ============================================================
# Prompt Utilities
# ============================================================

def compose_text_with_templates(text: str, templates=imagenet_templates):
    return [template.format(text) for template in templates]


def compose_text_for_ordinary_photo():
    return [
        "an ordinary photograph",
        "a normal photo",
        "a realistic photo",
        "a natural photograph",
        "a plain photo",
        "a typical photo",
        "a photographic image",
        "a standard photographic picture",
    ]


def use_custom_template(prompt_list):
    print("CUSTOM PROMPTS")
    return prompt_list


# ============================================================
# Image Normalization (LongCLIP)
# ============================================================

def normalize_image(image):
    """
    image: PIL.Image or torch image tensor convertible by preprocess
    """
    return _transform(image)#.unsqueeze(0)


def normalize_image_batch(images):
    return _transform(images)


# ============================================================
# Style Embedding (Single)
# ============================================================

def get_style_embedding(
    model,
    style_prompt=None,
    style_image=None,
    object_prompt="a Photo",
    device=DEVICE,
    adapter = None
):
    with torch.no_grad():

        # ----------------------------------------------------
        # Style feature (text or image)
        # ----------------------------------------------------
        if style_image is None:
            template_text = compose_text_with_templates(
                style_prompt, imagenet_templates_small
            )
            tokens = longclip.tokenize(template_text).to(device)
            style_features = model.encode_text(tokens)

            if adapter is not None:
                style_features = adapter(style_features)

            style_features = style_features.mean(dim=0, keepdim=True)
            style_features /= style_features.norm(dim=-1, keepdim=True)

        else:
            image = normalize_image(style_image).to(device)
            style_features = model.encode_image(image)
            if adapter is not None:
                style_features = adapter(style_features, modality = "image")
            style_features /= style_features.norm(dim=-1, keepdim=True)

        # ----------------------------------------------------
        # Source object feature
        # ----------------------------------------------------
        template_source = compose_text_with_templates(
            object_prompt, imagenet_templates_small
        )
        tokens_source = longclip.tokenize(template_source).to(device)

        text_source = model.encode_text(tokens_source)
        if adapter is not None:
            text_source = adapter(text_source)
        text_source = text_source.mean(dim=0, keepdim=True)
        text_source /= text_source.norm(dim=-1, keepdim=True)

        # ----------------------------------------------------
        # Style direction
        # ----------------------------------------------------
        style_direction = style_features - text_source
        style_direction /= style_direction.norm(dim=-1, keepdim=True)

        return style_direction, style_features


# ============================================================
# Style Embeddings (Batch)
# ============================================================

def get_style_embeddings_batch(
    model,
    style_prompts=None,
    style_images=None,
    object_prompt="a Photo",
    device=DEVICE,
    adapter = None
):
    """
    Returns:
        style_directions_text,
        text_feats,
        style_directions_imgs,
        img_feats
    """

    with torch.no_grad():

        # ----------------------------------------------------
        # 1) Encode all text prompts (batched)
        # ----------------------------------------------------
        all_texts = []
        for p in style_prompts:
            all_texts.extend(
                compose_text_with_templates(p, imagenet_templates_small)
            )

        tokens = longclip.tokenize(all_texts).to(device)
        text_feats = model.encode_text(tokens)
        if adapter is not None:
            text_feats = adapter(text_feats)


        text_feats = text_feats.view(
            len(style_prompts), -1, text_feats.size(-1)
        )
        text_feats = text_feats.mean(dim=1)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        # ----------------------------------------------------
        # 2) Encode all images (batched)
        # ----------------------------------------------------
        images_batch = normalize_image_batch(style_images).to(device)
        img_feats = model.encode_image(images_batch)
        if adapter is not None:
            img_feats = adapter(img_feats, modality = "image")
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        # ----------------------------------------------------
        # 3) Encode source object prompt
        # ----------------------------------------------------
        template_source = compose_text_with_templates(
            object_prompt, imagenet_templates_small
        )
        tokens_source = longclip.tokenize(template_source).to(device)

        src_feat = model.encode_text(tokens_source)
        if adapter is not None:
            src_feat = adapter(src_feat)
        src_feat = src_feat.mean(dim=0, keepdim=True)
        src_feat = src_feat / src_feat.norm(dim=-1, keepdim=True)

        src_expanded = src_feat.expand(img_feats.size(0), -1)

        # ----------------------------------------------------
        # 4) Compute style directions
        # ----------------------------------------------------
        style_directions_imgs = img_feats - src_expanded
        style_directions_imgs = (
            style_directions_imgs
            / style_directions_imgs.norm(dim=-1, keepdim=True)
        )

        style_directions_text = text_feats - src_expanded
        style_directions_text = (
            style_directions_text
            / style_directions_text.norm(dim=-1, keepdim=True)
        )

        return (
            style_directions_text,
            text_feats,
            style_directions_imgs,
            img_feats,
        )
