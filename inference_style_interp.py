import os
import sys
from math import floor
from pathlib import Path

import imageio
import torch
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from src.config import load_typed_root_config
from src.global_cfg import set_cfg
from src.misc.image_io import save_image
from src.misc.nn_module_tools import convert_to_buffer
from src.model.model import get_model
from src.utils.clip_utils import (
    get_style_embedding,
    load_image,
)
from src.utils.clip_utils import (
    load_model as load_clip_model,
)
from src.utils.image import (
    prepare_image_for_dino_patches,
    process_image,
)

# ------------------------------------------------------------
# Rendering (first view only)
# ------------------------------------------------------------


def render_first_view(intrinsics, extrinsics, h, w, decoder, gaussians):

    output = decoder.forward(
        gaussians,
        extrinsics[:, :1],
        intrinsics[:, :1].float(),
        torch.tensor([[0.1]], device=extrinsics.device),
        torch.tensor([[100.0]], device=extrinsics.device),
        (h, w),
    )

    rgb = output.color[0, 0].clamp(0.0, 1.0)
    return rgb


# ------------------------------------------------------------
# Style helpers
# ------------------------------------------------------------


def compute_style_dir(style_cfg, model, clip_model, device):
    if style_cfg.type == "text":
        with torch.autocast("cuda", dtype=torch.float32):
            clip_dir, clip_feat = get_style_embedding(
                clip_model,
                style_prompt=style_cfg.value,
                adapter=model.encoder.text_adapter,
            )
        return (clip_dir, None, None, clip_feat)

    elif style_cfg.type == "image":
        img = load_image(style_cfg.value).to(device)
        img_dino = prepare_image_for_dino_patches(img).unsqueeze(0)

        dino_embed = None
        if model.encoder.cond_type == "cross_attention_dino":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pkg = model.encoder.aggregator(
                    images=img_dino,
                    forward_mode="package_only",
                )
            dino_embed = pkg[0][:, pkg[-1] :, :]

        with torch.autocast("cuda", dtype=torch.float32):
            clip_dir, clip_feat = get_style_embedding(
                clip_model,
                style_image=img,
                adapter=model.encoder.text_adapter,
            )

        return (clip_dir, dino_embed, img_dino.shape, clip_feat)

    else:
        raise ValueError(f"Unknown style type: {style_cfg.type}")


def interpolate_style_dir(a, b, t):

    clip_a_dir, dino_a, shape_a, clip_a_target_feat = a
    clip_b_dir, dino_b, shape_b, clip_b_target_feat = b

    clip_dir = (1 - t) * clip_a_dir + t * clip_b_dir
    clip_target_feat = (1 - t) * clip_a_target_feat + t * clip_b_target_feat

    if dino_a is not None and dino_b is not None:
        dino = (1 - t) * dino_a + t * dino_b
        shape = shape_a
    else:
        dino, shape = None, None

    return (clip_dir, dino, shape, clip_target_feat)


# ------------------------------------------------------------
# Video helper
# ------------------------------------------------------------


def save_video(frames, path, fps=8):
    frames_np = [(f.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8") for f in frames]
    imageio.mimwrite(path, frames_np, fps=fps, quality=8)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------


def main(cfg_path: str, sample: int | None):
    cfg = OmegaConf.load(cfg_path)

    run_dir = Path(cfg.run_dir)
    hydra_cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    typed_cfg = load_typed_root_config(hydra_cfg)
    set_cfg(hydra_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = get_model(typed_cfg.model.encoder, typed_cfg.model.decoder)

    if cfg.ckpt:
        ckpt = cfg.ckpt
    else:
        ckpt = sorted((run_dir / "checkpoints").glob("*.ckpt"))[-1]
    state = torch.load(ckpt, map_location=device)["state_dict"]
    model.load_state_dict(
        {k.replace("model.", ""): v for k, v in state.items()},
        strict=False,
    )
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    # CLIP
    clip_model = load_clip_model()
    convert_to_buffer(clip_model, persistent=False)
    valid_exts = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")

    # Scenes
    for scene_name, scene in cfg.scenes.items():
        print(f"\n=== Scene: {scene_name} ===")
        img_folder = Path(scene.path)
        imgs_list = [f for f in os.listdir(img_folder) if f.lower().endswith(valid_exts)]

        total_images = len(imgs_list)
        final_sample = None
        print(f"Total images: {total_images}")

        # By default limit number of images to 20 if not specified to avoid long processing
        if total_images > 20 and sample is None:
            final_sample = 20

        # If sample has been specified, then sample uniformly from whole sequence
        if final_sample is not None:
            total_images = len(imgs_list)
            step = floor(total_images / final_sample)
            imgs_list = sorted([os.path.join(img_folder, f) for f in imgs_list])[::step]
        else:
            imgs_list = sorted([os.path.join(img_folder, f) for f in imgs_list])
        imgs = [process_image(p) for p in imgs_list]
        imgs = torch.stack(imgs, dim=0).unsqueeze(0).to(device=device)

        _, _, _, h, w = imgs.shape
        out_dir = Path(scene.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        rgb_dir = out_dir / "rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True)

        for pair in cfg.pairs:
            print(f"Interpolating pair: {pair.name}")

            style_a = compute_style_dir(pair.style_a, model, clip_model, device)
            style_b = compute_style_dir(pair.style_b, model, clip_model, device)

            frames = []
            ts = torch.linspace(0, 1, cfg.interpolation.num_steps)

            for t in tqdm(ts, desc="Interpolation progress"):
                style_t = interpolate_style_dir(style_a, style_b, t.item())

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    gaussians, ctx = model.inference(
                        (imgs + 1) * 0.5,
                        style_dir=style_t,
                    )

                rgb = render_first_view(
                    ctx["intrinsic"],
                    ctx["extrinsic"],
                    h,
                    w,
                    model.decoder,
                    gaussians,
                )

                # save immediately
                frame_idx = len(frames)
                img_path = rgb_dir / f"{pair.name}_im{frame_idx:03d}.jpg"
                save_image(rgb, img_path)

                # append for video
                frames.append(rgb)

            video_path = out_dir / f"{pair.name}_style_interp.mp4"
            save_video(frames, video_path, fps=cfg.interpolation.fps)

            print(f"Saved → {video_path}")

    print("All interpolations complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--sample", type=int, nargs="?", default=None, help="(Optional) Number of samples to use during inference"
    )
    args = parser.parse_args()
    main(args.config, args.sample)
