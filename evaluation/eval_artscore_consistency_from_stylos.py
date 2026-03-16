import sys

sys.path.append("evaluation/raft_core")

import argparse
import glob
import os
import re
from collections import defaultdict

import lpips
import numpy as np
import raft_core.softsplat as softsplat
import torch
from PIL import Image
from raft_core.raft import RAFT
from raft_core.utils.utils import InputPadder
from torch import nn
from torchvision import models, transforms
from tqdm import tqdm

##########################################################
# Backwarp Helper
##########################################################
backwarp_tenGrid = {}


def backwarp(tenIn, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = (
            torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3])
            .view(1, 1, 1, -1)
            .repeat(1, 1, tenFlow.shape[2], 1)
        )
        tenVer = (
            torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2])
            .view(1, 1, -1, 1)
            .repeat(1, 1, 1, tenFlow.shape[3])
        )
        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1).cuda()

    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenIn.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenIn.shape[2] - 1.0) / 2.0),
        ],
        1,
    )

    return torch.nn.functional.grid_sample(
        input=tenIn,
        grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


##########################################################
# Utilities
##########################################################
DEVICE = "cuda"


def load_image(imfile):
    img = np.array(Image.open(imfile).convert("RGB")).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


##########################################################
# STYLE-AWARE IMAGE LOADING (NEW, MINIMAL)
##########################################################
def load_images_grouped_by_style(directory):
    """
    Reads images named like: 11_12.jpg
    Returns: {style_id: [sorted frame paths]}
    """
    extensions = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    images = []

    for ext in extensions:
        images.extend(glob.glob(os.path.join(directory, ext)))

    style_dict = defaultdict(list)

    for path in images:
        name = os.path.basename(path)
        match = re.match(r"(\d+)_(\d+)", name)
        if match is None:
            continue
        style_id = int(match.group(1))
        frame_id = int(match.group(2))
        style_dict[style_id].append((frame_id, path))

    for style_id in style_dict:
        style_dict[style_id] = [p for _, p in sorted(style_dict[style_id], key=lambda x: x[0])]

    return dict(style_dict)


##########################################################
# Flow-based Evaluation Functions (UNCHANGED)
##########################################################
def evaluate_frame_pair(model, image1, image2, loss_fn_vgg, alpha=10, plot=False):
    padder = InputPadder(image1.shape)
    image1_padded, image2_padded = padder.pad(image1, image2)

    _, flow_up = model(image1_padded, image2_padded, iters=32, test_mode=True)

    tenMetric = torch.nn.functional.l1_loss(
        input=image1_padded, target=backwarp(image2_padded, flow_up), reduction="none"
    ).mean([1], True)

    tenSoftmax = softsplat.softsplat(
        tenIn=image1_padded, tenFlow=flow_up, tenMetric=(-alpha * tenMetric).clip(-alpha, alpha), strMode="soft"
    )

    mask = torch.where(tenSoftmax.sum(1, keepdim=True) == 0.0, 0.0, 1.0)
    lpips_score = loss_fn_vgg(tenSoftmax, image2_padded * mask)
    rmse_score = torch.sqrt(
        torch.nn.functional.mse_loss(tenSoftmax / 255.0, image2_padded * mask / 255.0, reduction="mean")
    )
    #########
    # if plot:
    #     save_dir = "hardcoded/path/for/vis/debug/consistency"
    #     # os.makedirs(save_dir, exist_ok=True)

    #     # take first item in batch
    #     vis_soft = tenSoftmax[0].detach().clamp(0, 255) / 255.0
    #     vis_gt = (image2_padded * mask)[0].detach().clamp(0, 255) / 255.0

    #     # --- RMSE MAP (per-pixel) ---
    #     rmse_map = torch.sqrt(
    #         torch.mean((vis_soft - vis_gt) ** 2, dim=0, keepdim=True)
    #     )  # shape: [1, H, W]

    #     # normalize RMSE map for visualization
    #     rmse_map = rmse_map / (rmse_map.max() + 1e-8)
    #     rmse_map = rmse_map.repeat(3, 1, 1)  # grayscale -> RGB

    #     # concatenate horizontally (C, H, 3W)
    #     vis = torch.cat([image1_padded[0]/255.0, image2_padded[0]/255.0, vis_soft, vis_gt, rmse_map], dim=2)

    #     # time-sorted filename
    #     timestamp = int(time.time() * 1000)
    #     filename = f"{timestamp}_{uuid.uuid4().hex}_rmse{rmse_score}.png"
    #     path = os.path.join(save_dir, filename)

    #     # vutils.save_image(vis, path)

    #     # --- SAVE WITH LABELS USING MATPLOTLIB ---
    #     vis_np = vis.permute(1, 2, 0).cpu().numpy()

    #     plt.figure(figsize=(16, 4))
    #     plt.imshow(vis_np)
    #     plt.axis("off")

    #     labels = [
    #         "Image 1",
    #         "Image 2",
    #         "Warped img1",
    #         "Img2 masked",
    #         f"RMSE\n{rmse_score:.4f}",
    #     ]

    #     H, W, _ = vis_np.shape
    #     n_imgs = len(labels)
    #     w = W // n_imgs

    #     for i, label in enumerate(labels):
    #         plt.text(
    #             i * w + 10,
    #             25,
    #             label,
    #             fontsize=16,
    #             color="black",
    #             bbox=dict(
    #                 facecolor="white",
    #                 alpha=0.75,
    #                 edgecolor="none",
    #                 boxstyle="round,pad=0.3",
    #             ),
    #         )

    #     plt.savefig(path, bbox_inches="tight", pad_inches=0)
    #     plt.close()
    #########

    return lpips_score.item(), rmse_score.item()


def evaluate_short_range(model, image_paths, loss_fn_vgg):
    lpips_vals, rmse_vals = [], []
    for i in range(len(image_paths) - 1):
        img1, img2 = load_image(image_paths[i]), load_image(image_paths[i + 1])
        lp, rm = evaluate_frame_pair(model, img1, img2, loss_fn_vgg, plot=False)
        lpips_vals.append(lp)
        rmse_vals.append(rm)
    return np.mean(lpips_vals), np.mean(rmse_vals)


def evaluate_long_range(model, image_paths, loss_fn_vgg, gap=7):
    lpips_vals, rmse_vals = [], []
    for i in range(gap, len(image_paths)):
        img1, img2 = load_image(image_paths[i - gap]), load_image(image_paths[i])
        lp, rm = evaluate_frame_pair(model, img1, img2, loss_fn_vgg, plot=False)
        lpips_vals.append(lp)
        rmse_vals.append(rm)
    return np.mean(lpips_vals), np.mean(rmse_vals)


##########################################################
# Main Evaluation Loop (MINIMAL CHANGE)
##########################################################
def main_eval(args):
    raft_model = torch.nn.DataParallel(RAFT(args))
    raft_model.load_state_dict(torch.load(args.model))
    raft_model = raft_model.module.to(DEVICE).eval()
    loss_fn_vgg = lpips.LPIPS(net="vgg").cuda()

    artscore_ckpt = torch.load(args.artscore_model)
    artscore_model = models.resnet50()
    artscore_model.fc = nn.Sequential(
        nn.Linear(2048, 1000),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1000, 1),
    )
    artscore_model.load_state_dict(artscore_ckpt)
    artscore_model = artscore_model.to(DEVICE).eval()

    artscore_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    scene_results = []

    # ##########
    # save_dir = "hardcoded/path/for/vis/debug/consistency"
    # if os.path.exists(save_dir):
    #     shutil.rmtree(save_dir)
    # os.makedirs(save_dir, exist_ok=True)
    ##########

    with torch.no_grad():
        for scene_dir in sorted(os.listdir(args.base_dir)):
            scene_path = os.path.join(args.base_dir, scene_dir)
            if not os.path.isdir(scene_path):
                continue

            print(f"\n[Scene: {scene_dir}]")

            lp_s_list, rm_s_list, lp_l_list, rm_l_list = [], [], [], []
            scene_artscore_list = []

            # style-aware grouping
            style_groups = load_images_grouped_by_style(scene_path)

            for style_id, paths in tqdm(sorted(style_groups.items())):
                if len(paths) < 1:
                    continue

                # ---- ArtScore ----
                art_scores = []
                for img_path in paths:
                    image = Image.open(img_path).convert("RGB")
                    input_tensor = artscore_transform(image).unsqueeze(0).to(DEVICE)
                    score = artscore_model(input_tensor)[0].item()
                    art_scores.append(score)

                scene_artscore_list.append(np.mean(art_scores))

                # ---- Flow metrics (WITHIN STYLE ONLY) ----
                if args.mode in ["both", "short"] and len(paths) > 1:
                    lp_s, rm_s = evaluate_short_range(raft_model, paths, loss_fn_vgg)
                    lp_s_list.append(lp_s)
                    rm_s_list.append(rm_s)
                    print(lp_s, rm_s)

                if args.mode in ["both", "long"] and len(paths) > args.long_gap:
                    lp_l, rm_l = evaluate_long_range(raft_model, paths, loss_fn_vgg, gap=args.long_gap)
                    lp_l_list.append(lp_l)
                    rm_l_list.append(rm_l)
                    print(lp_l, rm_l)

            scene_results.append(
                (
                    scene_dir,
                    np.mean(lp_s_list) if lp_s_list else -100000000,
                    np.mean(rm_s_list) if rm_s_list else -100000000,
                    np.mean(lp_l_list) if lp_l_list else -100000000,
                    np.mean(rm_l_list) if rm_l_list else -100000000,
                    np.mean(scene_artscore_list) if scene_artscore_list else -100000000,
                )
            )

            print(f"partial results after {scene_dir}:")
            print(scene_results)

    csv_path = os.path.join(os.path.dirname(args.base_dir), "eval_results_consistency_artscore.csv")

    # with open(csv_path,"w",newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["Scene","LPIPS_short","RMSE_short","LPIPS_long","RMSE_long","ArtScore"])
    #     for row in scene_results:
    #         writer.writerow(row)

    # print(f"\n[INFO] Results saved to {csv_path}")

    ######### save convinient format
    txt_path = os.path.join(os.path.dirname(args.base_dir), "results_consistency_artscore.txt")

    scenes_order = ["Train", "Truck", "M60", "Garden"]

    short_lpips = {}
    short_rmse = {}
    long_lpips = {}
    long_rmse = {}
    artscore_map = {}

    with open(txt_path, "w") as f:
        for scene, lpips_short, rmse_short, lpips_long, rmse_long, artscore in scene_results:
            # save raw values
            short_lpips[scene] = lpips_short
            short_rmse[scene] = rmse_short
            long_lpips[scene] = lpips_long
            long_rmse[scene] = rmse_long
            artscore_map[scene] = artscore

            # original logging
            f.write(f"{scene}\n")
            f.write(f"lpips_short: {lpips_short}\n")
            f.write(f"rmse_short: {rmse_short}\n")
            f.write(f"lpips_long: {lpips_long}\n")
            f.write(f"rmse_long: {rmse_long}\n")
            f.write(f"artscore: {artscore}\n\n")

        # ---------- Short-range LaTeX ----------
        short_range_latex = (
            r"\our{} & " + " & ".join(f"{short_lpips[s]:.3f} & {short_rmse[s]:.3f}" for s in scenes_order) + " \\\\\n"
        )

        # ---------- Long-range LaTeX ----------
        long_range_latex = (
            r"\our{} & " + " & ".join(f"{long_lpips[s]:.3f} & {long_rmse[s]:.3f}" for s in scenes_order) + " \\\\\n"
        )

        # ---------- ArtScore LaTeX (empty FID columns) ----------
        artscore_latex = r"\our{} & " + " & ".join(f"{artscore_map[s]:.2f} & " for s in scenes_order) + "\\\\\n"

        # write LaTeX blocks
        f.write("=== LaTeX-ready rows ===\n\n")
        f.write("Short-range latex:\n")
        f.write(short_range_latex + "\n")
        f.write("Long-range latex:\n")
        f.write(long_range_latex + "\n")
        f.write("ArtScore latex:\n")
        f.write(artscore_latex)


##########################################################
# Argument Parsing
##########################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/raft-things.pth")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--alternate_corr", action="store_true")
    parser.add_argument("--base_dir", type=str, required=True, help="Folder with target (stylized) renders")
    parser.add_argument("--mode", choices=["short", "long", "both", "none"], default="both")
    parser.add_argument("--long_gap", type=int, default=7)
    parser.add_argument("--artscore_model", type=str, required=True)
    args = parser.parse_args()

    main_eval(args)
