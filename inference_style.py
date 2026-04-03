import gc
import json
import os
import sys
from math import floor
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_typed_root_config
from src.global_cfg import set_cfg
from src.misc.image_io import save_image, save_interpolated_video
from src.misc.nn_module_tools import convert_to_buffer
from src.model.model import get_model
from src.model.ply_export import export_ply
from src.utils.clip_utils import get_style_embedding, load_image
from src.utils.clip_utils import load_model as load_clip_model
from src.utils.image import prepare_image_for_dino_patches, process_image


def render_image(
    intrinsics, extrinsics, height, width, decoder, gaussians
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_frames = extrinsics.shape[1]

    output = decoder.forward(
        gaussians,
        extrinsics,
        intrinsics.float(),
        torch.ones(1, num_frames, device=extrinsics.device) * 0.1,  # near plane
        torch.ones(1, num_frames, device=extrinsics.device) * 100,  # far plane
        (height, width),
    )

    rgb = output.color[0].clip(min=0.0, max=1.0)
    depth = output.depth[0]

    # Normalize depth for visualization
    # to avoid `quantile() input tensor is too large`
    num_views = extrinsics.shape[1]
    depth_norm = (depth - depth[::num_views].quantile(0.01)) / (
        depth[::num_views].quantile(0.99) - depth[::num_views].quantile(0.01)
    )
    depth_norm = plt.cm.turbo(depth_norm.cpu().numpy())
    depth_colored = torch.from_numpy(depth_norm[..., :3]).permute(0, 3, 1, 2).to(depth.device)
    depth_colored = depth_colored.clip(min=0.0, max=1.0)

    return rgb, depth, depth_colored


def load_hydra_config(run_dir: Path):
    hydra_dir = run_dir / ".hydra"
    if not hydra_dir.exists():
        raise FileNotFoundError(f"No .hydra folder in {run_dir}")
    return OmegaConf.load(hydra_dir / "config.yaml")


def find_latest_checkpoint(ckpt_dir: Path):
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=os.path.getmtime)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    return ckpts[-1]


def _report_peak_vram(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.cuda.synchronize(device)
    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    print(
        f"\nPeak VRAM (PyTorch allocated): {peak_alloc / (1024**3):.2f} GiB\n"
        f"Peak VRAM (caching allocator reserved): {peak_reserved / (1024**3):.2f} GiB"
    )

def main(config_path: str, sample: int | None):
    cfg_yaml = OmegaConf.load(config_path)

    run_dir = Path(cfg_yaml.run_dir)
    hydra_cfg = load_hydra_config(run_dir)
    cfg = load_typed_root_config(hydra_cfg)
    set_cfg(hydra_cfg)

    if cfg_yaml.ckpt:
        ckpt_path = cfg_yaml.ckpt
    else:
        ckpt_path = find_latest_checkpoint(run_dir / "checkpoints")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    
    # cfg.model.encoder.use_style=False #quick fix if you want to check without stylization. Not parametrized on purpose
    model = get_model(cfg.model.encoder, cfg.model.decoder)
    finetune_ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = finetune_ckpt["state_dict"]
    clean_state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=False)

    del finetune_ckpt, state_dict, clean_state_dict
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model = model.to(device=device).eval()
    for p in model.parameters():
        p.requires_grad = False

    if cfg.model.encoder.use_style:
        clip_model = load_clip_model()
        convert_to_buffer(clip_model, persistent=False)
    else:
        clip_model = None

    _report_peak_vram(device)

    global_texts = cfg_yaml.styles_for_all_scenes.get("text", [])
    global_imgs = cfg_yaml.styles_for_all_scenes.get("image", [])

    for scene_name, scene in cfg_yaml.scenes.items():
        print(f"\n=== Scene: {scene_name} ===")
        img_folder = Path(scene.path)
        output_folder = Path(scene.output)
        text_styles = sorted(list(set(global_texts + scene.get("style_prompts", []))))
        img_styles = sorted(list(set(global_imgs + scene.get("img_prompts", []))))

        valid_exts = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")
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
        b, v, _, h, w = imgs.shape

        # ---Text styles---
        for prompt in text_styles:
            print(f"→ Text style: {prompt}")
            if cfg.model.encoder.use_style:
                with torch.autocast(device_type="cuda", dtype=torch.float32):
                    style_dir = get_style_embedding(clip_model, style_prompt=prompt, adapter=model.encoder.text_adapter)
                style_dir = (style_dir[0], None, None, style_dir[1])
            else:
                style_dir = None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                gaussians, ctx = model.inference((imgs + 1) * 0.5, style_dir=style_dir)

            # hardcoded :60 will limit length of text prompt in output name
            # it does not influence generated output
            rgb, _, color_depth = render_image(ctx["intrinsic"], ctx["extrinsic"], h, w, model.decoder, gaussians)
            for idx, image_path in enumerate(imgs_list):
                filename = image_path.split("/")[-1]
                file_path = output_folder / "rgb" / f"{prompt.replace(' ', '_')[:60]}_rgb_{filename}"
                save_image(rgb[idx], file_path)
                file_path = output_folder / "depth" / f"{prompt.replace(' ', '_')[:60]}_color_depth_{filename}"
                save_image(color_depth[idx], file_path)

            save_interpolated_video(
                ctx["extrinsic"],
                ctx["intrinsic"],
                b,
                h,
                w,
                gaussians,
                output_folder,
                model.decoder,
                suffix=f"_{prompt.replace(' ', '_')[:60]}",
            )
            export_ply(
                gaussians.means[0],
                gaussians.scales[0],
                gaussians.rotations[0],
                gaussians.harmonics[0],
                gaussians.opacities[0],
                output_folder / f"gaussians_{prompt.replace(' ', '_')[:60]}.ply",
            )

        # ---Image styles---
        for style_img in img_styles:
            print(f"→ Image style: {style_img}")

            if cfg.model.encoder.use_style:
                # Load style image and extract style embedding
                style_img_raw = load_image(style_img).to(device=device)
                style_img_dino = prepare_image_for_dino_patches(style_img_raw).unsqueeze(0)

                style_embed = None
                if model.encoder.cond_type == "cross_attention_dino":
                    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                        pkg = model.encoder.aggregator(images=style_img_dino, forward_mode="package_only")
                    style_embed = pkg[0][:, pkg[-1] :, :]

            
                with torch.autocast(device_type="cuda", dtype=torch.float32):
                    style_dir = get_style_embedding(
                        clip_model, style_image=style_img_raw, adapter=model.encoder.text_adapter
                    )
                style_dir = (style_dir[0], style_embed, style_img_dino.shape, style_dir[1])
            else:
                style_dir = None

            # Inference
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                gaussians, ctx = model.inference((imgs + 1) * 0.5, style_dir=style_dir)

            # Render and save images
            rgb, _, color_depth = render_image(ctx["intrinsic"], ctx["extrinsic"], h, w, model.decoder, gaussians)
            for idx, image_path in enumerate(imgs_list):
                filename = image_path.split("/")[-1]
                file_path = output_folder / "rgb" / f"{Path(style_img).stem}_rgb_{filename}"
                save_image(rgb[idx], file_path)
                file_path = output_folder / "depth" / f"{Path(style_img).stem}_color_depth_{filename}"
                save_image(color_depth[idx], file_path)

            # Render interpolated video
            save_interpolated_video(
                ctx["extrinsic"],
                ctx["intrinsic"],
                b,
                h,
                w,
                gaussians,
                output_folder,
                model.decoder,
                suffix=f"_{Path(style_img).stem}",
            )

            # Save model as PLY
            export_ply(
                gaussians.means[0],
                gaussians.scales[0],
                gaussians.rotations[0],
                gaussians.harmonics[0],
                gaussians.opacities[0],
                output_folder / f"gaussians_{Path(style_img).stem}.ply",
            )

    print("\nAll scenes complete.")
    _report_peak_vram(device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AnySplat inference using YAML config.")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--sample", type=int, nargs="?", default=None, help="(Optional) Number of samples to use during inference"
    )
    args = parser.parse_args()
    main(args.config, args.sample)
