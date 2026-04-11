#!/bin/bash
# Setup for YOLO universal encoder distillation (repos + deps + teacher weights)
# Run this first. Dataset download is separate (get_datacomp12m.sh).
#
# Teachers load via native Python (not TorchScript .ts) because EUPE/DINOv3 RoPE
# position encoding hardcodes device in torch.arange, making traced graphs non-portable.
set -e
cd /home/fatih/dev/ultralytics

echo "=== Step 1: Clone reference repos ==="
[ -d /home/fatih/dev/eupe ] || git clone https://github.com/facebookresearch/eupe /home/fatih/dev/eupe
[ -d /home/fatih/dev/RADIO ] || git clone https://github.com/NVlabs/RADIO /home/fatih/dev/RADIO
[ -d /home/fatih/dev/unic ] || git clone https://github.com/naver/unic /home/fatih/dev/unic
[ -d /home/fatih/dev/dune ] || git clone https://github.com/naver/dune /home/fatih/dev/dune

echo "=== Step 2: Install dependencies ==="
source .venv/bin/activate
uv pip install \
    "transformers==5.5.0" \
    "webdataset==0.2.111" \
    "img2dataset" \
    "huggingface_hub>=1.8.0"

echo "=== Step 3: Download EUPE weights (3 variants) ==="
python -c "
from huggingface_hub import hf_hub_download
for repo, fname in [
    ('facebook/EUPE-ViT-B', 'EUPE-ViT-B.pt'),
    ('facebook/EUPE-ViT-S', 'EUPE-ViT-S.pt'),
    ('facebook/EUPE-ConvNeXt-B', 'EUPE-ConvNeXt-B.pt'),
]:
    path = hf_hub_download(repo, fname)
    print(f'  {repo} -> {path}')
"

echo "=== Step 4: Pre-download DINOv3 + SigLIP2 weights ==="
python -c "
from transformers import AutoModel, SiglipVisionModel

for name in [
    'facebook/dinov3-vitb16-pretrain-lvd1689m',
    'facebook/dinov3-vitl16-pretrain-lvd1689m',
    'facebook/dinov3-convnext-base-pretrain-lvd1689m',
]:
    AutoModel.from_pretrained(name)
    print(f'  {name} OK')

SiglipVisionModel.from_pretrained('google/siglip2-giant-opt-patch16-384')
print('  google/siglip2-giant-opt-patch16-384 OK')
"

echo "=== Step 5: Verify all teachers ==="
python -c "
import torch
from ultralytics.nn.teacher_model import build_teacher_model, TEACHER_REGISTRY

# Skip vit7b (26GB, slow to load) and sam3:l (1024px, needs lots of RAM)
for variant in ['eupe:vitb16', 'eupe:vits16', 'eupe:convnextb',
                'dinov3:vitb16', 'dinov3:vitl16', 'dinov3:convnextb',
                'siglip2:g']:
    reg = TEACHER_REGISTRY[variant]
    teacher = build_teacher_model(variant, torch.device('cpu'))
    dummy = torch.randn(1, 3, reg['imgsz'], reg['imgsz'])
    with torch.no_grad():
        out = teacher.encode(dummy)
    cls_shape = out.cls.shape if out.cls is not None else None
    print(f'  {variant}: cls={cls_shape}, patches={out.patches.shape}')
    del teacher
print('All teachers verified OK')
"

echo "Setup complete! Run get_datacomp12m.sh next for dataset."
