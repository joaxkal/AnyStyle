import os
import sys
import time
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as tv_transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from src.config import load_typed_root_config
from src.global_cfg import set_cfg
from src.misc.image_io import save_image, save_interpolated_video
from src.misc.nn_module_tools import convert_to_buffer
from src.model.model import get_model
from src.model.ply_export import export_ply
from src.utils.clip_utils import get_style_embedding
from src.utils.clip_utils import load_model as load_clip_model
from src.utils.image import prepare_image_for_dino_patches


class StylOSDataset(Dataset):
    def __init__(self, content_root: Path, style_root: Path) -> None:
        self.style_files = sorted(list(style_root.iterdir()))
        self.scenes = []

        for scene in content_root.iterdir():
            image_files = sorted(scene.iterdir(), key=lambda p: int(p.stem))
            self.scenes.append({"name": scene.stem, "files": image_files})

        self.num_scenes = len(self.scenes)
        self.num_styles = len(self.style_files)
        self.num_examples = self.num_scenes

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx) -> tuple[str, torch.Tensor, list[torch.Tensor]]:
        # load content
        content = []
        scene = self.scenes[idx]

        for image_path in scene["files"]:  # [0:16]:
            content_image = Image.open(image_path).convert("RGB")
            content.append(self._transform_content(content_image))

        # load style
        style = []

        # suboptimal, we could load it to memory once
        for style_file in self.style_files:
            style_image = Image.open(style_file).convert("RGB")
            style.append(self._transform_style(style_image))

        return (scene["name"], torch.stack(content, dim=0), style)

    def _transform_content(self, image: Image) -> torch.Tensor:
        width, height = image.size
        if width > height:
            new_height = 448
            new_width = int(width * (new_height / height))
        else:
            new_width = 448
            new_height = int(height * (new_width / width))
        image = image.resize((new_width, new_height))

        # Center crop
        left = (new_width - 448) // 2
        top = (new_height - 448) // 2
        right = left + 448
        bottom = top + 448
        image = image.crop((left, top, right, bottom))
        image_tensor = tv_transforms.functional.pil_to_tensor(image).float()
        image_tensor = image_tensor / 255.0  # [0, 1]

        assert image_tensor.min() >= 0.0 and image_tensor.max() <= 1.0

        return image_tensor

    def _transform_style(self, image: Image, target_size=256) -> torch.Tensor:
        image = tv_transforms.functional.pil_to_tensor(image)[:3]  # -> (3, H, W)
        image = image.float() / 255.0

        assert image.min() >= 0.0 and image.max() <= 1.0

        _, H, W = image.shape

        if H < W:
            ratio = W / H
            new_H = target_size
            new_W = int(ratio * new_H)
        else:
            ratio = H / W
            new_W = target_size
            new_H = int(ratio * new_W)

        image = tv_transforms.functional.resize(image, [new_H, new_W], antialias=True)
        image = tv_transforms.functional.center_crop(image, [target_size, target_size])

        return image


class TextDataset(Dataset):
    def __init__(self, content_root: Path, style_path: Path) -> None:
        self.styles_descriptions = self._read_style_descriptions(style_path)
        self.style_files = sorted((style_path.parent / "style_images").iterdir())  # TODO better parametrization
        self.scenes = []  # TODO compare if description match image name

        for scene in content_root.iterdir():
            image_files = sorted(scene.iterdir(), key=lambda p: int(p.stem))
            self.scenes.append({"name": scene.stem, "files": image_files})

        self.num_scenes = len(self.scenes)
        self.num_styles = len(self.styles_descriptions)
        self.num_examples = self.num_scenes

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx) -> tuple[str, torch.Tensor, list[torch.Tensor]]:
        # load content
        content = []
        scene = self.scenes[idx]

        for image_path in scene["files"]:  # [0:16]:
            content_image = Image.open(image_path).convert("RGB")
            content.append(self._transform_content(content_image))

        # load style
        style = []

        # suboptimal, we could load it to memory once
        for style_file in self.style_files:
            style_image = Image.open(style_file).convert("RGB")
            style.append(self._transform_style(style_image))

        return (scene["name"], torch.stack(content, dim=0), self.styles_descriptions, style)

    def _transform_content(self, image: Image) -> torch.Tensor:
        width, height = image.size
        if width > height:
            new_height = 448
            new_width = int(width * (new_height / height))
        else:
            new_width = 448
            new_height = int(height * (new_width / width))
        image = image.resize((new_width, new_height))

        # Center crop
        left = (new_width - 448) // 2
        top = (new_height - 448) // 2
        right = left + 448
        bottom = top + 448
        image = image.crop((left, top, right, bottom))
        image_tensor = tv_transforms.functional.pil_to_tensor(image).float()
        image_tensor = image_tensor / 255.0  # [0, 1]

        assert image_tensor.min() >= 0.0 and image_tensor.max() <= 1.0

        return image_tensor

    def _transform_style(self, image: Image, target_size=256) -> torch.Tensor:
        image = tv_transforms.functional.pil_to_tensor(image)[:3]  # -> (3, H, W)
        image = image.float() / 255.0

        assert image.min() >= 0.0 and image.max() <= 1.0

        _, H, W = image.shape

        if H < W:
            ratio = W / H
            new_H = target_size
            new_W = int(ratio * new_H)
        else:
            ratio = H / W
            new_W = target_size
            new_H = int(ratio * new_W)

        image = tv_transforms.functional.resize(image, [new_H, new_W], antialias=True)
        image = tv_transforms.functional.center_crop(image, [target_size, target_size])

        return image

    def _read_style_descriptions(self, style_path: Path) -> list:
        style_descriptions = ["" for _ in range(50)]

        with style_path.open() as style_file:
            for line in style_file:
                splits = line.split(";", 1)
                file_name = splits[0].strip()
                description = splits[1].strip()

                idx = int(file_name.split(".")[0])
                assert idx <= 49
                style_descriptions[idx] = description

        return style_descriptions


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


def infere_from_style_image(
    content_path: Path,
    style_path: Path,
    model,
    clip_model,
    device,
    output_path: Path,
    save_video: bool = False,
    save_ply: bool = False,
) -> None:
    test_dataset = StylOSDataset(content_path, style_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    time_acc = 0.0
    time_counter = 0

    for scene_name, content_images, style_images in test_dataloader:
        scene_name = scene_name[0]
        print(f"Processing {scene_name}")

        # inference for each style
        for style_idx, style_image in tqdm(enumerate(style_images)):
            style_image = style_image.to(device)
            content_images = content_images.to(device)
            b, v, _, h, w = content_images.shape

            # Load style image and extract style embedding
            style_img_dino = prepare_image_for_dino_patches(style_image).unsqueeze(0)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                pkg = model.encoder.aggregator(images=style_img_dino, forward_mode="package_only")
            style_embed = pkg[0][:, pkg[-1] :, :]
            start_time = time.time()
            style_dir = get_style_embedding(clip_model, style_image=style_image)
            style_dir = (style_dir[0], style_embed, style_img_dino.shape, style_dir[1])

            # Inference
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                gaussians, ctx = model.inference(content_images, style_dir=style_dir)

            # Render and save images
            rgb, _, _ = render_image(ctx["intrinsic"], ctx["extrinsic"], h, w, model.decoder, gaussians)
            end_time = time.time()
            time_acc += end_time - start_time
            time_counter += 1

            content = content_images[0]

            # Save images with corresponding names for ArtFID
            for i in range(16):
                save_image(
                    content[i],
                    output_path / "content" / scene_name / f"{style_idx}_{i}.jpg",
                )
                save_image(
                    rgb[i],
                    output_path / "generated" / scene_name / f"{style_idx}_{i}.jpg",
                )
                save_image(
                    style_image,
                    output_path / "style" / scene_name / f"{style_idx}_{i}.jpg",
                )

            # Render interpolated video
            if save_video:
                video_root_path = output_path / "video" / scene_name
                if not video_root_path.exists():
                    video_root_path.mkdir(parents=True)

                save_interpolated_video(
                    ctx["extrinsic"],
                    ctx["intrinsic"],
                    b,
                    h,
                    w,
                    gaussians,
                    video_root_path,
                    model.decoder,
                    suffix=f"_scene_{style_idx}",
                    save_depth=False,
                )

            # Save model as PLY
            if save_ply:
                ply_root_path = output_path / "ply"
                if not ply_root_path.exists():
                    ply_root_path.mkdir(parents=True)

                export_ply(
                    gaussians.means[0],
                    gaussians.scales[0],
                    gaussians.rotations[0],
                    gaussians.harmonics[0],
                    gaussians.opacities[0],
                    ply_root_path / scene_name / f"gaussians_scene_{style_idx}.ply",
                )

    with (output_path / "time.txt").open("w") as file:
        file.write(f"{time_acc / time_counter}")


def infere_from_style_text(
    content_path: Path,
    style_path: Path,
    model,
    clip_model,
    device,
    output_path: Path,
    save_video: bool = False,
    save_ply: bool = False,
) -> None:
    test_dataset = TextDataset(content_path, style_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    time_acc = 0.0
    time_counter = 0

    for scene_name, content_images, styles_desc, style_images in test_dataloader:
        scene_name = scene_name[0]  # Unpack -> pytorch collate_fn
        print(f"Processing {scene_name}")

        content_images = content_images.to(device)
        b, v, _, h, w = content_images.shape

        for style_idx, desc in tqdm(enumerate(styles_desc)):
            desc = desc[0]  # Unpack -> pytorch collate_fn
            print(f"→ Description: {desc}")
            start_time = time.time()
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                style_dir = get_style_embedding(clip_model, style_prompt=desc, adapter=model.encoder.text_adapter)
            style_dir = (style_dir[0], None, None, style_dir[1])

            # Inference
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                gaussians, ctx = model.inference(content_images, style_dir=style_dir)

            # Render and save images
            rgb, _, _ = render_image(ctx["intrinsic"], ctx["extrinsic"], h, w, model.decoder, gaussians)
            end_time = time.time()
            time_acc += end_time - start_time
            time_counter += 1

            content = content_images[0]

            # Save images with corresponding names
            for i in range(16):
                save_image(
                    content[i],
                    output_path / "content" / scene_name / f"{style_idx}_{i}.jpg",
                )
                save_image(
                    rgb[i],
                    output_path / "generated" / scene_name / f"{style_idx}_{i}.jpg",
                )
                save_image(
                    style_images[style_idx],
                    output_path / "style" / scene_name / f"{style_idx}_{i}.jpg",
                )

            # Render interpolated video
            if save_video:
                video_root_path = output_path / "video" / scene_name
                if not video_root_path.exists():
                    video_root_path.mkdir(parents=True)

                save_interpolated_video(
                    ctx["extrinsic"],
                    ctx["intrinsic"],
                    b,
                    h,
                    w,
                    gaussians,
                    video_root_path,
                    model.decoder,
                    suffix=f"_scene_{style_idx}",
                    save_depth=False,
                )

            # Save model as PLY
            if save_ply:
                ply_root_path = output_path / "ply"
                if not ply_root_path.exists():
                    ply_root_path.mkdir(parents=True)

                export_ply(
                    gaussians.means[0],
                    gaussians.scales[0],
                    gaussians.rotations[0],
                    gaussians.harmonics[0],
                    gaussians.opacities[0],
                    ply_root_path / scene_name / f"gaussians_scene_{style_idx}.ply",
                )

    with (output_path / "time.txt").open("w") as file:
        file.write(f"{time_acc / time_counter}")


def main(
    run_dir: str,
    content_path: str,
    style_path: str,
    output_path: str,
    save_video: bool = False,
    save_ply: bool = False,
    ckpt: str = None,
):
    # load config
    run_dir = Path(run_dir)
    hydra_cfg = load_hydra_config(run_dir)
    cfg = load_typed_root_config(hydra_cfg)
    set_cfg(hydra_cfg)

    # load CLIP model
    clip_model = load_clip_model()
    convert_to_buffer(clip_model, persistent=False)

    # load main model
    if ckpt is None:
        ckpt_path = find_latest_checkpoint(run_dir / "checkpoints")
    else:
        ckpt_path = ckpt
    print("Evaluating ckpt: ", ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg.model.encoder, cfg.model.decoder)
    finetune_ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = finetune_ckpt["state_dict"]
    clean_state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=False)
    model = model.to(device=device)
    model.eval()

    if os.path.isdir(style_path):
        print("Running with image conditioning.")
        infere_from_style_image(
            Path(content_path),
            Path(style_path),
            model,
            clip_model,
            device,
            Path(output_path),
            save_video,
            save_ply,
        )
    else:
        print("Running with text conditioning.")
        infere_from_style_text(
            Path(content_path),
            Path(style_path),
            model,
            clip_model,
            device,
            Path(output_path),
            save_video,
            save_ply,
        )

    print("\nAll scenes complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AnySplat inference")
    parser.add_argument("--run_dir", type=str, help="Path to run folder to evaluate")
    parser.add_argument("--content_path", type=str, help="Path to content images")
    parser.add_argument("--style_path", type=str, help="Path to style images")
    parser.add_argument("--output_path", type=str, help="Path to stylized renders")
    parser.add_argument("--save_video", default=False, action="store_true")
    parser.add_argument("--save_ply", default=False, action="store_true")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to selected ckpt.pth")

    args = parser.parse_args()
    main(args.run_dir, args.content_path, args.style_path, args.output_path, args.save_video, args.save_ply, args.ckpt)
