# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This file provides the definition of the convolutional heads used to predict masks, as well as the losses
"""
import io
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from PIL import Image

import util.box_ops as box_ops
from util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list

try:
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    pass


class DETRsegm(nn.Module):
    def __init__(self, detr: nn.Module, freeze_detr: bool = False):
        super().__init__()
        self.detr = detr

        if freeze_detr:
            for p in self.parameters():
                p.requires_grad_(False)

        hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim)

    def forward(self, samples: NestedTensor) -> Dict[str, Tensor]:
        """검출 출력에 "pred_masks": (B, num_queries, H/4, W/4)를 추가해서 돌려준다.

        DETR.forward와 마찬가지로, 반환 애노테이션은 TorchScript가 보는 타입이다.
        aux_loss가 켜져 있으면 "aux_outputs" 키가 더 붙는다.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.detr.backbone(samples)

        bs = features[-1].tensors.shape[0]

        src, mask = features[-1].decompose()
        assert mask is not None, "DETRsegm needs the padding mask produced by the backbone"
        assert len(features) == 4, \
            f"the mask head needs all 4 backbone stages; build the backbone with "\
            f"return_interm_layers=True (got {len(features)})"
        src_proj = self.detr.input_proj(src)
        hs, memory = self.detr.transformer(src_proj, mask, self.detr.query_embed.weight, pos[-1])

        outputs_class = self.detr.class_embed(hs)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.detr.aux_loss:
            out['aux_outputs'] = self.detr._set_aux_loss(outputs_class, outputs_coord)

        # FIXME h_boxes takes the last one computed, keep this in mind
        bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)

        seg_masks = self.mask_head(src_proj, bbox_mask, [features[2].tensors, features[1].tensors, features[0].tensors])
        outputs_seg_masks = seg_masks.view(bs, self.detr.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])
        assert outputs_seg_masks.dim() == 4, \
            f"expected (B, num_queries, H, W), got {outputs_seg_masks.shape}"

        out["pred_masks"] = outputs_seg_masks
        return out


def _expand(tensor: Tensor, length: int) -> Tensor:
    """(B, C, H, W) -> (B * length, C, H, W). 각 배치 원소를 length번 복제해 이어붙인다.

    쿼리마다 피처맵 사본이 하나씩 필요하므로 배치 축에 쿼리를 접어 넣는 것.
    """
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim: int, fpn_dims: List[int], context_dim: int):
        assert len(fpn_dims) == 3, f"expected 3 FPN levels, got {len(fpn_dims)}"
        # 채널이 context_dim//2 ... //16으로 반씩 줄고 각 단계에 GroupNorm(8, C)가 붙는다.
        # 따라서 가장 좁은 단계 context_dim//16까지 8로 나누어떨어져야 한다 -> 128의 배수.
        assert context_dim % 128 == 0, \
            f"context_dim must be a multiple of 128 (it is halved 4 times and each step "\
            f"uses GroupNorm with 8 groups), got {context_dim}"
        assert dim % 8 == 0, f"dim must be a multiple of 8 for GroupNorm, got {dim}"
        super().__init__()

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        self.dim = dim

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]) -> Tensor:
        """Args:
            x         : (B, context_dim, H, W)               -- input_proj를 통과한 피처
            bbox_mask : (B, num_queries, nheads, H, W)       -- 쿼리별 어텐션 맵
            fpns      : layer3, layer2, layer1 피처 (해상도가 점점 커지는 순서)
        Returns:
            (B * num_queries, 1, H', W')  -- 쿼리 축이 배치에 접혀 있다. 부르는 쪽에서 편다.
        """
        assert x.dim() == 4, f"expected x (B, C, H, W), got {x.shape}"
        assert bbox_mask.dim() == 5, \
            f"expected bbox_mask (B, num_queries, nheads, H, W), got {bbox_mask.shape}"
        assert bbox_mask.shape[0] == x.shape[0], \
            f"batch mismatch: x {x.shape[0]} vs bbox_mask {bbox_mask.shape[0]}"
        assert len(fpns) == 3, f"expected 3 FPN levels, got {len(fpns)}"

        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)

        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        x = self.out_lay(x)
        assert x.shape[1] == 1, f"mask head should emit a single channel, got {x.shape[1]}"
        return x


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim: int, hidden_dim: int, num_heads: int,
                 dropout: float = 0.0, bias: bool = True):
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q: Tensor, k: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Args:
            q    : (B, num_queries, query_dim)  -- 디코더 출력
            k    : (B, query_dim, H, W)         -- 인코더 memory를 2D로 되돌린 것
            mask : (B, H, W)                    bool, True = 패딩
        Returns:
            (B, num_queries, num_heads, H, W)  -- value를 곱하지 않은 어텐션 확률 그 자체
        """
        assert q.dim() == 3, f"expected q (B, num_queries, C), got {q.shape}"
        assert k.dim() == 4, f"expected k (B, C, H, W), got {k.shape}"
        assert q.shape[0] == k.shape[0], f"batch mismatch: q {q.shape[0]} vs k {k.shape[0]}"
        if mask is not None:
            assert mask.shape[-2:] == k.shape[-2:], \
                f"mask {mask.shape} must match the feature plane of k {k.shape}"

        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        assert weights.dim() == 5, \
            f"expected (B, num_queries, num_heads, H, W), got {weights.shape}"
        return weights


def dice_loss(inputs: Tensor, targets: Tensor, num_boxes: float) -> Tensor:
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    assert inputs.shape == targets.shape, \
        f"inputs {tuple(inputs.shape)} and targets {tuple(targets.shape)} must match"
    assert num_boxes > 0, f"num_boxes is the loss normalizer and must be positive, got {num_boxes}"
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs: Tensor, targets: Tensor, num_boxes: float,
                       alpha: float = 0.25, gamma: float = 2) -> Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    assert inputs.shape == targets.shape, \
        f"inputs {tuple(inputs.shape)} and targets {tuple(targets.shape)} must match"
    assert num_boxes > 0, f"num_boxes is the loss normalizer and must be positive, got {num_boxes}"
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class PostProcessSegm(nn.Module):
    def __init__(self, threshold: float = 0.5):
        assert 0.0 < threshold < 1.0, f"threshold is a probability, got {threshold}"
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results: List[Dict[str, Tensor]], outputs: Dict[str, Tensor],
                orig_target_sizes: Tensor, max_target_sizes: Tensor) -> List[Dict[str, Tensor]]:
        """예측 마스크를 패딩 전 크기로 잘라내고 원본 이미지 크기로 되돌린다."""
        assert len(orig_target_sizes) == len(max_target_sizes), \
            f"{len(orig_target_sizes)} original sizes but {len(max_target_sizes)} padded sizes"
        assert len(results) == len(orig_target_sizes), \
            f"{len(results)} detection results but {len(orig_target_sizes)} images"
        assert "pred_masks" in outputs, f"outputs has no 'pred_masks': {sorted(outputs)}"
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        outputs_masks = outputs["pred_masks"].squeeze(2)
        outputs_masks = F.interpolate(outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False)
        outputs_masks = (outputs_masks.sigmoid() > self.threshold).cpu()

        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]["masks"] = F.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            ).byte()

        return results


class PostProcessPanoptic(nn.Module):
    """This class converts the output of the model to the final panoptic result, in the format expected by the
    coco panoptic API """

    def __init__(self, is_thing_map: Dict[int, bool], threshold: float = 0.85):
        """
        Parameters:
           is_thing_map: This is a whose keys are the class ids, and the values a boolean indicating whether
                          the class is  a thing (True) or a stuff (False) class
           threshold: confidence threshold: segments with confidence lower than this will be deleted
        """
        assert 0.0 < threshold < 1.0, f"threshold is a probability, got {threshold}"
        assert len(is_thing_map) > 0, "is_thing_map must not be empty"
        super().__init__()
        self.threshold = threshold
        self.is_thing_map = is_thing_map

    def forward(self, outputs: Dict[str, Tensor],
                processed_sizes: Union[List[Tuple[int, int]], Tensor],
                target_sizes: Optional[Union[List[Tuple[int, int]], Tensor]] = None
                ) -> List[Dict[str, Any]]:
        """ This function computes the panoptic prediction from the model's predictions.
        Parameters:
            outputs: This is a dict coming directly from the model. See the model doc for the content.
            processed_sizes: This is a list of tuples (or torch tensors) of sizes of the images that were passed to the
                             model, ie the size after data augmentation but before batching.
            target_sizes: This is a list of tuples (or torch tensors) corresponding to the requested final size
                          of each prediction. If left to None, it will default to the processed_sizes
            """
        if target_sizes is None:
            target_sizes = processed_sizes
        assert len(processed_sizes) == len(target_sizes)
        out_logits, raw_masks, raw_boxes = outputs["pred_logits"], outputs["pred_masks"], outputs["pred_boxes"]
        assert len(out_logits) == len(raw_masks) == len(target_sizes)
        preds = []

        def to_tuple(tup: Union[Tuple[int, int], Tensor]) -> Tuple[int, int]:
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())

        for cur_logits, cur_masks, cur_boxes, size, target_size in zip(
            out_logits, raw_masks, raw_boxes, processed_sizes, target_sizes
        ):
            # we filter empty queries and detection below threshold
            scores, labels = cur_logits.softmax(-1).max(-1)
            keep = labels.ne(outputs["pred_logits"].shape[-1] - 1) & (scores > self.threshold)
            cur_scores, cur_classes = cur_logits.softmax(-1).max(-1)
            cur_scores = cur_scores[keep]
            cur_classes = cur_classes[keep]
            cur_masks = cur_masks[keep]
            cur_masks = interpolate(cur_masks[:, None], to_tuple(size), mode="bilinear").squeeze(1)
            cur_boxes = box_ops.box_cxcywh_to_xyxy(cur_boxes[keep])

            h, w = cur_masks.shape[-2:]
            assert len(cur_boxes) == len(cur_classes)

            # It may be that we have several predicted masks for the same stuff class.
            # In the following, we track the list of masks ids for each stuff class (they are merged later on)
            cur_masks = cur_masks.flatten(1)
            stuff_equiv_classes = defaultdict(lambda: [])
            for k, label in enumerate(cur_classes):
                if not self.is_thing_map[label.item()]:
                    stuff_equiv_classes[label.item()].append(k)

            def get_ids_area(masks: Tensor, scores: Tensor,
                             dedup: bool = False) -> Tuple[List[int], Image.Image]:
                # This helper function creates the final panoptic segmentation image
                # It also returns the area of the masks that appears on the image

                m_id = masks.transpose(0, 1).softmax(-1)

                if m_id.shape[-1] == 0:
                    # We didn't detect any mask :(
                    m_id = torch.zeros((h, w), dtype=torch.long, device=m_id.device)
                else:
                    m_id = m_id.argmax(-1).view(h, w)

                if dedup:
                    # Merge the masks corresponding to the same stuff class
                    for equiv in stuff_equiv_classes.values():
                        if len(equiv) > 1:
                            for eq_id in equiv:
                                m_id.masked_fill_(m_id.eq(eq_id), equiv[0])

                final_h, final_w = to_tuple(target_size)

                seg_img = Image.fromarray(id2rgb(m_id.view(h, w).cpu().numpy()))
                seg_img = seg_img.resize(size=(final_w, final_h), resample=Image.NEAREST)

                np_seg_img = (
                    torch.ByteTensor(torch.ByteStorage.from_buffer(seg_img.tobytes())).view(final_h, final_w, 3).numpy()
                )
                m_id = torch.from_numpy(rgb2id(np_seg_img))

                area = []
                for i in range(len(scores)):
                    area.append(m_id.eq(i).sum().item())
                return area, seg_img

            area, seg_img = get_ids_area(cur_masks, cur_scores, dedup=True)
            if cur_classes.numel() > 0:
                # We know filter empty masks as long as we find some
                while True:
                    filtered_small = torch.as_tensor(
                        [area[i] <= 4 for i, c in enumerate(cur_classes)], dtype=torch.bool, device=keep.device
                    )
                    if filtered_small.any().item():
                        cur_scores = cur_scores[~filtered_small]
                        cur_classes = cur_classes[~filtered_small]
                        cur_masks = cur_masks[~filtered_small]
                        area, seg_img = get_ids_area(cur_masks, cur_scores)
                    else:
                        break

            else:
                cur_classes = torch.ones(1, dtype=torch.long, device=cur_classes.device)

            segments_info = []
            for i, a in enumerate(area):
                cat = cur_classes[i].item()
                segments_info.append({"id": i, "isthing": self.is_thing_map[cat], "category_id": cat, "area": a})
            del cur_classes

            with io.BytesIO() as out:
                seg_img.save(out, format="PNG")
                predictions = {"png_string": out.getvalue(), "segments_info": segments_info}
            preds.append(predictions)
        return preds


def _demo() -> None:
    """파놉틱/세그멘테이션 헤드를 굴려보는 예시. 실행: python -m models.segmentation"""
    torch.manual_seed(0)

    hidden_dim, nheads, num_queries = 256, 8, 4
    B, H, W = 1, 8, 12

    print("=" * 66)
    print("MHAttentionMap -- value를 곱하지 않고 어텐션 확률만 돌려준다")
    print("=" * 66)
    attn = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
    attn.eval()
    q = torch.randn(B, num_queries, hidden_dim)      # 디코더 출력
    k = torch.randn(B, hidden_dim, H, W)             # 인코더 memory (2D로 되돌린 것)
    mask = torch.zeros(B, H, W, dtype=torch.bool)
    mask[0, :, W // 2:] = True                       # 오른쪽 절반은 패딩

    with torch.no_grad():
        weights = attn(q, k, mask=mask)
    print(f"  q {tuple(q.shape)}, k {tuple(k.shape)}")
    print(f"  -> {tuple(weights.shape)}  = (B, num_queries, nheads, H, W)")

    assert weights.shape == (B, num_queries, nheads, H, W)
    # 주의: softmax는 flatten(2) 뒤에 걸리므로 "헤드 x 공간" 전체에 대해 한 번 정규화된다.
    # 헤드마다 따로 1이 되는 게 아니다.
    total = weights.sum(dim=(2, 3, 4))
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5), \
        "softmax는 (헤드, H, W)를 한 덩어리로 정규화한다"
    per_head = weights.sum(dim=(3, 4))
    assert not torch.allclose(per_head, torch.ones_like(per_head), atol=1e-3), \
        "헤드별로는 1이 되지 않는다"
    print(f"  [ok] 확률 합은 (헤드 x 공간) 전체에서 1  -- 헤드 하나당은 약 {1 / nheads:.3f}")
    assert (weights[0, :, :, :, W // 2:] == 0).all(), "패딩 위치에는 확률이 0이어야 한다"
    print("  [ok] 패딩 위치의 확률은 정확히 0 (softmax 전에 -inf로 채워짐)")

    print()
    print("=" * 66)
    print("MaskHeadSmallConv -- FPN 방식으로 해상도를 되살리는 마스크 헤드")
    print("=" * 66)
    fpn_dims = [1024, 512, 256]                      # resnet50의 layer3, layer2, layer1
    head = MaskHeadSmallConv(hidden_dim + nheads, fpn_dims, hidden_dim)
    head.eval()
    src_proj = torch.randn(B, hidden_dim, H, W)
    fpns = [torch.randn(B, 1024, H * 2, W * 2),
            torch.randn(B, 512, H * 4, W * 4),
            torch.randn(B, 256, H * 8, W * 8)]
    with torch.no_grad():
        seg = head(src_proj, weights, fpns)
    print(f"  입력 피처 {tuple(src_proj.shape)} @ {H}x{W}")
    print(f"  FPN 단계 {[tuple(f.shape[-2:]) for f in fpns]}")
    print(f"  -> {tuple(seg.shape)}  = (B*num_queries, 1, H', W')")

    assert seg.shape == (B * num_queries, 1, H * 8, W * 8), \
        "가장 고해상도 FPN 단계까지 올라간다"
    print("  [ok] 쿼리 축이 배치에 접혀 나온다 -- 쿼리마다 마스크를 하나씩 그리기 때문")

    # _expand가 바로 그 '접기'를 한다.
    expanded = _expand(src_proj, num_queries)
    assert expanded.shape == (B * num_queries, hidden_dim, H, W)
    assert torch.equal(expanded[0], expanded[1]), "같은 이미지의 사본이므로 동일해야 한다"
    print(f"  [ok] _expand: {tuple(src_proj.shape)} -> {tuple(expanded.shape)} (사본 복제)")

    print()
    print("=" * 66)
    print("dice_loss / sigmoid_focal_loss -- 마스크용 손실 두 가지")
    print("=" * 66)
    targets = (torch.rand(3, 64) > 0.5).float()
    perfect = torch.where(targets > 0.5, 12.0, -12.0)     # 로짓: 맞추면 큰 값
    wrong = -perfect

    d_good = dice_loss(perfect, targets, num_boxes=3).item()
    d_bad = dice_loss(wrong, targets, num_boxes=3).item()
    f_good = sigmoid_focal_loss(perfect, targets, num_boxes=3).item()
    f_bad = sigmoid_focal_loss(wrong, targets, num_boxes=3).item()
    print(f"  dice_loss          맞춤 {d_good:.6f}  <->  틀림 {d_bad:.6f}")
    print(f"  sigmoid_focal_loss 맞춤 {f_good:.6f}  <->  틀림 {f_bad:.6f}")

    assert d_good < 1e-4 and f_good < 1e-4, "완벽히 맞추면 두 손실 모두 0에 가깝다"
    assert d_bad > 0.9, "완전히 틀리면 dice는 1에 가까워진다"
    assert f_bad > f_good
    print("  [ok] 맞추면 0, 틀리면 커진다")

    # focal loss는 '쉬운 예제'의 기여를 (1-p_t)^gamma로 눌러버린다.
    easy = torch.where(targets > 0.5, 4.0, -4.0)
    hard = torch.where(targets > 0.5, 0.2, -0.2)
    f_easy = sigmoid_focal_loss(easy, targets, num_boxes=3).item()
    f_hard = sigmoid_focal_loss(hard, targets, num_boxes=3).item()
    assert f_hard > f_easy * 10, "애매한 예제가 훨씬 큰 손실을 받아야 한다"
    print(f"  [ok] 확신하는 예제 {f_easy:.5f} << 애매한 예제 {f_hard:.5f} (focal의 목적)")

    print()
    print("=" * 66)
    print("DETRsegm -- 검출 모델을 감싸서 pred_masks를 덧붙인다")
    print("=" * 66)
    detr = _build_tiny_detr_for_segm(hidden_dim, nheads, num_queries)
    segm_model = DETRsegm(detr, freeze_detr=True)
    segm_model.eval()

    images = torch.randn(1, 3, 64, 96)
    img_mask = torch.zeros(1, 64, 96, dtype=torch.bool)
    with torch.no_grad():
        out = segm_model(NestedTensor(images, img_mask))

    print(f"  입력 {tuple(images.shape)}")
    for k in ("pred_logits", "pred_boxes", "pred_masks"):
        print(f"  {k:12s} {tuple(out[k].shape)}")

    assert out["pred_masks"].shape[:2] == (1, num_queries)
    # 마스크는 layer1(stride 4) 해상도까지 올라온다.
    assert out["pred_masks"].shape[-2:] == (64 // 4, 96 // 4)
    assert all(not p.requires_grad for p in detr.parameters()), \
        "freeze_detr=True면 검출 쪽은 전부 얼어야 한다"
    print("  [ok] 마스크는 쿼리마다 한 장씩, 입력의 1/4 해상도로 나온다")
    print("  [ok] freeze_detr=True -> 검출 파라미터는 모두 동결")

    print()
    print("모든 검사 통과.")


def _build_tiny_detr_for_segm(hidden_dim: int, nheads: int, num_queries: int) -> nn.Module:
    """데모용 DETR. 마스크 헤드가 resnet50의 채널 수를 가정하므로 resnet50을 쓰되,
    가중치는 내려받지 않고 무작위로 초기화한다 (weights=None)."""
    import torchvision

    from .backbone import BackboneBase, FrozenBatchNorm2d, Joiner
    from .detr import DETR
    from .position_encoding import PositionEmbeddingSine
    from .transformer import Transformer

    resnet = torchvision.models.resnet50(weights=None, norm_layer=FrozenBatchNorm2d)
    backbone = Joiner(
        # 마스크 헤드는 네 단계를 모두 필요로 한다.
        BackboneBase(resnet, train_backbone=False, num_channels=2048, return_interm_layers=True),
        PositionEmbeddingSine(num_pos_feats=hidden_dim // 2, normalize=True),
    )
    backbone.num_channels = 2048
    transformer = Transformer(d_model=hidden_dim, nhead=nheads, num_encoder_layers=1,
                              num_decoder_layers=1, dim_feedforward=128,
                              return_intermediate_dec=True)
    return DETR(backbone, transformer, num_classes=5, num_queries=num_queries)


if __name__ == "__main__":
    _demo()
