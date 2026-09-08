# DETR hands-on tutorial

A guided walk through **DETR** (*End-to-End Object Detection with Transformers*, Carion et al. 2020) for people who are new to PyTorch and new to object detection.

Every notebook runs against **this repository's own source** — not `torch.hub`, not a re-implementation. When you edit `models/detr.py`, the notebooks change with it.

The paper is at [`../references/2005.12872v3.pdf`](../references/2005.12872v3.pdf). Each notebook cites the exact section it is explaining.

---

## Setup

The environment is managed with [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync
```

That creates `.venv/` with torch, torchvision, scipy, matplotlib and a Jupyter kernel.

**Running the notebooks**

- **VS Code** — open any `.ipynb`, then pick the kernel `detr/.venv/bin/python` (top right → *Select Kernel* → *Python Environments*).
- **Browser** — `uv run --with jupyterlab jupyter lab`

Optional, only for COCO mAP evaluation (needs a C compiler):

```bash
uv sync --extra coco
```

### What gets downloaded

On first run the notebooks fetch the pretrained DETR-R50 checkpoint (~159 MB, cached in `~/.cache/torch/`) and a few COCO sample images (cached in `tutorial/_assets/`, gitignored). Everything after that is offline.

---

## The notebooks

Read them in order — each builds on the last.

| # | Notebook | What you'll learn | Paper |
|---|---|---|---|
| 01 | [Why DETR, and a first detection](01_why_detr_and_first_detection.ipynb) | Detection as **direct set prediction**; why anchors and NMS disappear; run the pretrained model | §1, Fig. 1 |
| 02 | [Backbone, shapes, positional encoding](02_backbone_shapes_and_positional_encoding.ipynb) | How a photo becomes a **sequence**; padding masks; sine position encodings. **Heavy on tensor shapes.** | §3.2 |
| 03 | [Transformer and object queries](03_transformer_and_object_queries.ipynb) | Encoder, decoder, and what an **object query** actually is | §3.2, Fig. 2, 7 |
| 04 | [Hungarian matching and the set loss](04_hungarian_matching_and_set_loss.ipynb) | **The core idea of the paper** — how to compute a loss on an unordered set | §3.1, Eq. 1–2 |
| 05 | [Attention visualization](05_attention_visualization.ipynb) | What the model looks at; reproduce the paper's attention figures | §4.2, Fig. 3, 6 |
| 06 | [Fine-tune on your own data](06_finetune_on_your_own_data.ipynb) | Train it yourself, end to end, in ~2 minutes | §3.2, §4 |

`detr_utils.py` holds the shared boilerplate (COCO class names, plotting, model loading) so the notebooks stay focused. It also puts the repo root on `sys.path`, which is why `from models.detr import DETR` works from inside `tutorial/`.

---

## Notes for PyTorch beginners

**Read the shape comments.** Every tensor is annotated like this:

```python
src = proj.flatten(2).permute(2, 0, 1)   # shape: (B, C, H, W) -> (H*W, B, C)
```

Shapes are the fastest way to understand a vision model. If you only skim one thing, skim those.

**Two conventions that trip people up:**

- **Channels-first.** PyTorch images are `(C, H, W)`, not `(H, W, C)`.
- **Sequence-first.** This DETR predates `batch_first=True`, so transformer tensors are `(sequence, batch, channels)` — batch is in the *middle*.

**Hardware.** Everything runs on CPU, CUDA, or Apple Silicon (MPS); `get_device()` picks automatically. Notebooks 01–05 are inference only (seconds). Notebook 06 trains for ~2.5 minutes on an M-series Mac.

---

## Things worth knowing about the codebase

- **`num_classes` is a misnomer.** It means `max_class_id + 1`, not the number of classes. COCO's ids run to 90, so DETR passes 91. See the comment at [`models/detr.py:304`](../models/detr.py#L304).
- **Feature-map sizes round up.** A 1066px width becomes 34 feature columns, not 33. Use `ceil(size / 32)`.
- **`torchvision` deprecation warnings** about `pretrained=` are expected — this is 2020 code on a modern torchvision. It still works correctly.
- **Notebooks are gitignored by default** in this repo (`*.ipynb`). `.gitignore` has an explicit exception for `tutorial/**/*.ipynb` so this tutorial is tracked.
