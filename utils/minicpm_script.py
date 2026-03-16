import os
import random

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# -------------------------
#  MODEL INIT
# -------------------------
torch.manual_seed(100)

model_name = "openbmb/MiniCPM-V-4_5"
model = AutoModel.from_pretrained(
    model_name, trust_remote_code=True, attn_implementation="sdpa", torch_dtype=torch.bfloat16
)
model = model.eval().to(device="cuda:1")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# -------------------------
#  CONFIG
# -------------------------
root_folder = "/path/to/datasets/wikiart"
output_file = "/path/to/datasets/wikiart/img_descriptions_long.txt"

prompt = "In 300 chars, give a precise description of image style visual features: exact color and brightness specs, contrast, contour line shape/weight, and brushwork or texture patterns."


# -------------------------
#  COLLECT IMAGE PATHS FIRST
# -------------------------
valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


processed_paths = set()
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if ";" in line:
                rel = line.split(";")[0].strip()
                processed_paths.add(rel)

image_paths = []
for root, _, files in os.walk(root_folder):
    for f in files:
        if f.lower().endswith(valid_ext):
            full = os.path.join(root, f)
            rel = os.path.relpath(full, root_folder)
            if rel not in processed_paths:
                image_paths.append(full)

random.shuffle(image_paths)


# -------------------------
#  PROCESS FUNCTION
# -------------------------
def process_image(img_path):
    try:
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        msgs = [{"role": "user", "content": [img, prompt]}]
        answer = model.chat(msgs=msgs, tokenizer=tokenizer, enable_thinking=False, use_fast=True)
        return answer.strip().replace("\n", " ")
    except Exception as e:
        return f"[error: {e}]"


# -------------------------
#  RUN WITH PROGRESS BAR
# -------------------------
with open(output_file, "a", encoding="utf-8") as fout:
    for full_path in tqdm(image_paths, desc="Processing images"):
        rel_path = os.path.relpath(full_path, root_folder)
        desc = process_image(full_path)
        fout.write(f"{rel_path};{desc}\n")

print("\nDONE — saved descriptions to", output_file)
