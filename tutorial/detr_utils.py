"""
Shared helpers for the DETR tutorial notebooks.

Keeping boilerplate here (class names, plotting, model loading) lets each
notebook stay focused on one idea. Import it with:

    from detr_utils import *
"""
import os
import sys

# The notebooks live in tutorial/, but the DETR source lives one level up.
# Put the repo root on sys.path so `from models.detr import DETR` works.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

ASSETS = os.path.join(REPO_ROOT, "tutorial", "_assets")
os.makedirs(ASSETS, exist_ok=True)

import argparse

import matplotlib.pyplot as plt
import requests
import torch
import torchvision.transforms as T
from PIL import Image

from models.backbone import build_backbone
from models.detr import DETR, MLP, PostProcess, SetCriterion
from models.matcher import HungarianMatcher
from models.transformer import build_transformer

# ---------------------------------------------------------------------------
# COCO classes
# ---------------------------------------------------------------------------
# DETR predicts 91 "classes" + 1 no-object slot = 92 logits per query.
# COCO's category ids are not contiguous (they run 1..90 with gaps), so the
# gaps are filled with 'N/A' placeholders. Index 0 is unused.
# The list MUST be exactly 91 long -- one short and every label after the gap
# is silently wrong.
COCO_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush',
]
assert len(COCO_CLASSES) == 91, f"expected 91 classes, got {len(COCO_CLASSES)}"

# Colors for drawing boxes (cycled).
COLORS = [
    [0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933],
]

SAMPLE_IMAGES = {
    "cats":     "http://images.cocodataset.org/val2017/000000039769.jpg",
    "street":   "http://images.cocodataset.org/val2017/000000000139.jpg",
    "horses":   "http://images.cocodataset.org/val2017/000000006471.jpg",
    "kitchen":  "http://images.cocodataset.org/val2017/000000002153.jpg",
}

DETR_R50_URL = "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
# DETR's eval transform: resize shortest side to 800px, to tensor, ImageNet
# normalize. Note there is NO fixed crop -- DETR handles variable input sizes.
NORMALIZE = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
default_transform = T.Compose([T.Resize(800), T.ToTensor(), NORMALIZE])


def get_device():
    """CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def detr_args(**overrides):
    """The argparse.Namespace that build_backbone/build_transformer expect.

    Defaults match the released DETR-R50 checkpoint. dropout=0.0 because we
    only do inference in these notebooks (dropout is a no-op under .eval()
    anyway, but 0.0 makes the intent explicit).
    """
    args = argparse.Namespace(
        backbone="resnet50", dilation=False, position_embedding="sine",
        lr_backbone=0, masks=False,
        hidden_dim=256, dropout=0.0, nheads=8, dim_feedforward=2048,
        enc_layers=6, dec_layers=6, pre_norm=False,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def build_detr(num_classes=91, num_queries=100, aux_loss=False, **overrides):
    """Assemble DETR from this repo's own modules (not torch.hub)."""
    args = detr_args(**overrides)
    return DETR(
        build_backbone(args), build_transformer(args),
        num_classes=num_classes, num_queries=num_queries, aux_loss=aux_loss,
    )


def load_pretrained_detr(device=None, aux_loss=False):
    """Build DETR-R50 and load the official COCO weights. Returns eval() model."""
    model = build_detr(aux_loss=aux_loss)
    ck = torch.hub.load_state_dict_from_url(DETR_R50_URL, map_location="cpu")
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()
    if device is not None:
        model.to(device)
    return model


def load_image(name_or_url):
    """Load a sample image by nickname, URL, or local path. Cached in _assets/."""
    url = SAMPLE_IMAGES.get(name_or_url, name_or_url)
    if os.path.exists(url):
        return Image.open(url).convert("RGB")
    path = os.path.join(ASSETS, os.path.basename(url))
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(requests.get(url, timeout=60).content)
    return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------
def box_cxcywh_to_xyxy(b):
    """(cx, cy, w, h) -> (x0, y0, x1, y1). Works on the last dim."""
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def rescale_bboxes(boxes, size):
    """Normalized cxcywh in [0,1] -> absolute xyxy pixels. `size` is PIL (W, H)."""
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(boxes)
    return b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(pil_img, prob, boxes, ax=None, title=None, linewidth=2.5):
    """Draw detections.

    prob  : (n_keep, 91) softmax probabilities WITHOUT the no-object column
    boxes : (n_keep, 4)  absolute xyxy pixel coordinates
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(pil_img)
    for i, (p, (xmin, ymin, xmax, ymax)) in enumerate(zip(prob, boxes.tolist())):
        c = COLORS[i % len(COLORS)]
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=linewidth))
        cl = p.argmax()
        ax.text(xmin, ymin, f'{COCO_CLASSES[cl]}: {p[cl]:0.2f}',
                fontsize=11, bbox=dict(facecolor=c, alpha=0.6, edgecolor='none'),
                color='white')
    ax.axis('off')
    if title:
        ax.set_title(title)
    return ax


@torch.no_grad()
def detect(model, pil_img, device=None, threshold=0.9):
    """Run DETR on a PIL image.

    Returns (probs_kept, boxes_kept_xyxy_pixels, raw_outputs, keep_mask).
    """
    device = device or next(model.parameters()).device
    x = default_transform(pil_img).unsqueeze(0).to(device)   # (1, 3, H, W)
    outputs = model(x)
    # drop the last column: it is the "no object" class
    probs = outputs['pred_logits'].softmax(-1)[0, :, :-1].cpu()   # (100, 91)
    keep = probs.max(-1).values > threshold
    boxes = rescale_bboxes(outputs['pred_boxes'][0, keep].cpu(), pil_img.size)
    return probs[keep], boxes, outputs, keep
