# DETR hands-on tutorial

A guided walk through **DETR** (*End-to-End Object Detection with Transformers*, Carion et al. 2020) for people who are new to PyTorch and new to object detection.

You start by building attention from scratch, and you finish by writing DETR yourself in 50 lines and loading the real pretrained weights into your own class.

If PyTorch itself is new, start with [`00 · PyTorch essentials`](00_pytorch_essentials.ipynb) — every operation DETR uses, each tied back to the line of `models/` that needs it.

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

On first run the notebooks fetch pretrained weights (cached in `~/.cache/torch/`) and a few COCO sample images (cached in `tutorial/_assets/`, gitignored):

| File | Size | Used by |
|---|---|---|
| DETR-R50 checkpoint | 159 MB | `02`–`07` |
| Faster R-CNN (for the NMS comparison) | 160 MB | `02` only |
| Mini-DETR checkpoint | 159 MB | `08` only |

Everything after that is offline. Notebooks `00` and `01` download nothing at all — they run on CPU with tensors they make up on the spot.

---

## The notebooks

Read them in order — each builds on the last. `00` doubles as a standalone reference: dip back into it whenever a line of PyTorch is in your way.

| # | Notebook | What you'll learn | Paper |
|---|---|---|---|
| 00 | [PyTorch essentials](00_pytorch_essentials.ipynb) | **Start here if PyTorch is new.** The ~25 operations DETR actually uses — shape surgery, broadcasting, advanced indexing, `nn.Module`, autograd, `state_dict` — each one grounded in a real line of this repo | — |
| 01 | [Attention from scratch](01_attention_from_scratch.ipynb) | **Start here if transformers are new.** Q/K/V, softmax, multi-head — built by hand and checked against PyTorch. Ends with *permutation invariance*, which explains everything that follows | App. A.1 |
| 02 | [Why DETR, and a first detection](02_why_detr_and_first_detection.ipynb) | Detection as **direct set prediction**. Includes a side-by-side with Faster R-CNN showing what NMS is actually for | §1, Fig. 1 |
| 03 | [Backbone, shapes, positional encoding](03_backbone_shapes_and_positional_encoding.ipynb) | How a photo becomes a **sequence**; padding masks; sine position encodings. **Heavy on tensor shapes** | §3.2 |
| 04 | [Transformer and object queries](04_transformer_and_object_queries.ipynb) | Encoder, decoder, and what an **object query** really is | §3.2, Fig. 2, 7 |
| 05 | [Hungarian matching and the set loss](05_hungarian_matching_and_set_loss.ipynb) | **The core idea of the paper** — how to compute a loss on an unordered set | §3.1, Eq. 1–2 |
| 06 | [Attention visualization](06_attention_visualization.ipynb) | What the model looks at; reproduces the paper's attention figures | §4.2, Fig. 3, 6 |
| 07 | [Fine-tune on your own data](07_finetune_on_your_own_data.ipynb) | Train it yourself, end to end, in ~2.5 minutes | §3.2, §4 |
| 08 | [**Capstone:** build DETR from scratch](08_build_detr_from_scratch.ipynb) | Write the whole model in 50 lines, load the official weights into it, detect | §3.2 |

Every notebook ends with **exercises that have worked solutions** in collapsible `<details>` blocks. Try them before opening the answer — they're where the understanding actually sticks.

`detr_utils.py` holds the shared boilerplate (COCO class names, plotting, model loading) so the notebooks stay focused. It also puts the repo root on `sys.path`, which is why `from models.detr import DETR` works from inside `tutorial/`.

### If you're short on time

- **Just want to use DETR?** `02` → `08`.
- **Want to understand the paper's contribution?** `02` → `05`.
- **Want to train on your own data?** `02` → `03` → `07`.
- **Transformers already familiar?** Skip `01`.
- **PyTorch not yet familiar?** Start at `00` — it is the only notebook with no prerequisites.

---

## Notes for PyTorch beginners

**Start with `00`.** [`00 · PyTorch essentials`](00_pytorch_essentials.ipynb) is a reference as much as a lecture: a lookup table at the top maps each operation to the line of `models/` that needs it, so you can read it straight through or jump to whatever has you stuck. It needs no downloads and runs in seconds on CPU.

**Read the shape comments.** Every tensor is annotated like this:

```python
src = proj.flatten(2).permute(2, 0, 1)   # shape: (B, C, H, W) -> (H*W, B, C)
```

Shapes are the fastest way to understand a vision model. If you only skim one thing, skim those.

**Two conventions that trip people up:**

- **Channels-first.** PyTorch images are `(C, H, W)`, not `(H, W, C)`.
- **Sequence-first.** This DETR predates `batch_first=True`, so transformer tensors are `(sequence, batch, channels)` — batch is in the *middle*.

**Hardware.** Everything runs on CPU, CUDA, or Apple Silicon (MPS); `get_device()` picks automatically. Notebooks `00`–`06` and `08` are inference only (seconds). Notebook `07` trains for ~2.5 minutes on an M-series Mac.

---

## Things worth knowing about the codebase

- **`num_classes` is a misnomer.** It means `max_class_id + 1`, not the number of classes. COCO's ids run to 90, so DETR passes 91. See the comment at [`models/detr.py:304`](../models/detr.py#L304).
- **Feature-map sizes round up.** A 1066px width becomes 34 feature columns, not 33. Use `ceil(size / 32)`, never `// 32`.
- **The COCO class list must be exactly 91 long.** One short and every label after the gap is silently wrong.
- **`torchvision` deprecation warnings** about `pretrained=` are expected — this is 2020 code on a modern torchvision. It still works correctly.
- **Notebooks are gitignored by default** in this repo (`*.ipynb`). `.gitignore` has an explicit exception for `tutorial/**/*.ipynb` so this tutorial is tracked.
