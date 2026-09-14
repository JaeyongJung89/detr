# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import argparse
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone: nn.Module, transformer: nn.Module, num_classes: int,
                 num_queries: int, aux_loss: bool = False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        assert num_classes > 0, f"num_classes must be positive, got {num_classes}"
        assert num_queries > 0, f"num_queries must be positive, got {num_queries}"
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor) -> Dict[str, Tensor]:
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.

            반환 애노테이션이 Dict[str, Tensor]인 것은 TorchScript가 보는 타입이다.
            aux_loss가 켜져 있으면 파이썬에서는 "aux_outputs" 키에 리스트가 하나 더
            붙으므로 실제 값 타입은 균일하지 않다. _set_aux_loss의 주석 참고.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None, "DETR needs the padding mask produced by the backbone"
        assert src.dim() == 4, f"expected backbone feature (B, C, H, W), got {src.shape}"

        # hs: (num_decoder_layers, B, num_queries, hidden_dim) -- 레이어마다 한 장씩
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        assert hs.dim() == 4, \
            f"expected (num_layers, B, num_queries, hidden_dim), got {hs.shape}"
        assert hs.shape[1] == src.shape[0], \
            f"batch mismatch: {src.shape[0]} images vs {hs.shape[1]} decoder outputs"
        assert hs.shape[2] == self.num_queries, \
            f"decoder returned {hs.shape[2]} queries, expected {self.num_queries}"

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        assert outputs_coord.shape[-1] == 4, \
            f"boxes must be 4 numbers (cx, cy, w, h), got {outputs_coord.shape[-1]}"
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class: Tensor, outputs_coord: Tensor):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        #
        # 반환 타입은 List[Dict[str, Tensor]] -- 디코더 레이어마다 한 dict씩, 마지막
        # 레이어는 제외(그건 이미 out에 들어 있다). 일부러 애노테이션을 달지 않았다:
        # @torch.jit.unused는 본문만 지우고 시그니처는 남겨서, 타입을 적으면
        # TorchScript가 Dict[str, Tensor]인 out에 리스트를 넣는다고 보고 컴파일을
        # 거부한다. 애노테이션이 없으면 값 타입을 Tensor로 추론해 통과한다.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes: int, matcher: nn.Module, weight_dict: Dict[str, float],
                 eos_coef: float, losses: List[str]):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        assert num_classes > 0, f"num_classes must be positive, got {num_classes}"
        assert eos_coef > 0, f"eos_coef must be positive, got {eos_coef}"
        assert len(losses) > 0, "at least one loss must be requested"
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]],
                    indices: List[Tuple[Tensor, Tensor]], num_boxes: float,
                    log: bool = True) -> Dict[str, Tensor]:
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]

        Shapes:
            outputs["pred_logits"] : (B, num_queries, num_classes + 1)
            indices                : 배치마다 (pred_idx, tgt_idx) 한 쌍
        """
        assert 'pred_logits' in outputs, f"outputs has no 'pred_logits': {sorted(outputs)}"
        src_logits = outputs['pred_logits']
        assert src_logits.dim() == 3, \
            f"expected (B, num_queries, num_classes+1), got {tuple(src_logits.shape)}"
        assert src_logits.shape[-1] == self.num_classes + 1, \
            f"classifier emits {src_logits.shape[-1]} logits, expected num_classes+1 = {self.num_classes + 1}"
        assert len(targets) == src_logits.shape[0] == len(indices), \
            f"batch mismatch: {src_logits.shape[0]} predictions, {len(targets)} targets, {len(indices)} matches"
        # self.num_classes가 no-object 슬롯의 id다. 실제 레이블이 그보다 크거나 같으면
        # cross_entropy가 범위를 벗어난다. build()의 num_classes 주석 참고.
        for i, t in enumerate(targets):
            assert t["labels"].numel() == 0 or int(t["labels"].max()) < self.num_classes, \
                f"target {i} has label id {int(t['labels'].max())}, but num_classes is "\
                f"{self.num_classes} (ids must be < num_classes; {self.num_classes} is the no-object slot)"

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]],
                         indices: List[Tuple[Tensor, Tensor]], num_boxes: float) -> Dict[str, Tensor]:
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]],
                   indices: List[Tuple[Tensor, Tensor]], num_boxes: float) -> Dict[str, Tensor]:
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs, f"outputs has no 'pred_boxes': {sorted(outputs)}"
        assert num_boxes > 0, f"num_boxes is the loss normalizer and must be positive, got {num_boxes}"
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # 매칭이 1:1이므로 짝지어진 박스 개수와 좌표 수가 정확히 같아야 한다.
        assert src_boxes.shape == target_boxes.shape, \
            f"matched boxes disagree: predictions {tuple(src_boxes.shape)} vs targets {tuple(target_boxes.shape)}"
        assert src_boxes.shape[-1] == 4, \
            f"boxes must be (cx, cy, w, h), got last dim {src_boxes.shape[-1]}"

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]],
                   indices: List[Tuple[Tensor, Tensor]], num_boxes: float) -> Dict[str, Tensor]:
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs, f"outputs has no 'pred_masks': {sorted(outputs)}"

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        assert src_masks.shape == target_masks.shape, \
            f"mask shapes disagree: {tuple(src_masks.shape)} vs {tuple(target_masks.shape)}"
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        """매칭 결과를 outputs[batch_idx, query_idx] 형태의 advanced-indexing 키로 편다."""
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        """위와 같은 일을 타깃 쪽 인덱스에 대해 한다."""
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss: str, outputs: Dict[str, Any], targets: List[Dict[str, Tensor]],
                 indices: List[Tuple[Tensor, Tensor]], num_boxes: float,
                 **kwargs: Any) -> Dict[str, Tensor]:
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs: Dict[str, Any], targets: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        assert 'pred_logits' in outputs and 'pred_boxes' in outputs, \
            f"outputs must carry 'pred_logits' and 'pred_boxes', got {sorted(outputs)}"
        assert len(targets) == outputs['pred_logits'].shape[0], \
            f"{outputs['pred_logits'].shape[0]} predictions but {len(targets)} targets"
        for i, t in enumerate(targets):
            assert 'labels' in t and 'boxes' in t, \
                f"target {i} must carry 'labels' and 'boxes', got {sorted(t)}"
            assert len(t['labels']) == len(t['boxes']), \
                f"target {i}: {len(t['labels'])} labels but {len(t['boxes'])} boxes"

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs: Dict[str, Tensor], target_sizes: Tensor) -> List[Dict[str, Tensor]]:
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert out_logits.dim() == 3, \
            f"expected pred_logits (B, num_queries, C), got {tuple(out_logits.shape)}"
        assert out_bbox.shape[:2] == out_logits.shape[:2], \
            f"pred_boxes {tuple(out_bbox.shape)} does not line up with pred_logits {tuple(out_logits.shape)}"
        assert out_bbox.shape[-1] == 4, f"boxes must be 4 numbers, got {out_bbox.shape[-1]}"
        assert len(out_logits) == len(target_sizes), \
            f"{len(out_logits)} predictions but {len(target_sizes)} target sizes"
        assert target_sizes.shape[1] == 2, \
            f"target_sizes must be (B, 2) as (height, width), got {tuple(target_sizes.shape)}"

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        assert len(results) == len(target_sizes), \
            f"produced {len(results)} results for {len(target_sizes)} images"
        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        assert num_layers >= 1, f"an MLP needs at least one layer, got {num_layers}"
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: Tensor) -> Tensor:
        """마지막 층에는 활성화를 붙이지 않는다 (여기선 박스 좌표를 그대로 뽑아야 하므로)."""
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args: argparse.Namespace) -> Tuple[nn.Module, nn.Module, Dict[str, nn.Module]]:
    """Returns: (model, criterion, postprocessors)"""
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors


def _build_tiny_detr(num_classes: int = 5, num_queries: int = 4,
                     hidden_dim: int = 32, aux_loss: bool = False) -> DETR:
    """데모용 초소형 DETR. 사전학습 가중치를 받지 않도록 resnet18을 새로 만든다."""
    import torchvision

    from .backbone import BackboneBase, FrozenBatchNorm2d, Joiner
    from .position_encoding import PositionEmbeddingSine
    from .transformer import Transformer

    resnet = torchvision.models.resnet18(weights=None, norm_layer=FrozenBatchNorm2d)
    backbone = Joiner(
        BackboneBase(resnet, train_backbone=False, num_channels=512, return_interm_layers=False),
        PositionEmbeddingSine(num_pos_feats=hidden_dim // 2, normalize=True),
    )
    backbone.num_channels = 512
    transformer = Transformer(d_model=hidden_dim, nhead=4, num_encoder_layers=2,
                              num_decoder_layers=2, dim_feedforward=64,
                              return_intermediate_dec=True)
    return DETR(backbone, transformer, num_classes=num_classes,
                num_queries=num_queries, aux_loss=aux_loss)


def _demo() -> None:
    """DETR 전체 파이프라인을 굴려보는 예시. 실행: python -m models.detr"""
    torch.manual_seed(0)

    num_classes, num_queries = 5, 4
    B, H, W = 2, 64, 128

    print("=" * 66)
    print("DETR -- 이미지를 넣으면 항상 num_queries개의 예측이 나온다")
    print("=" * 66)
    model = _build_tiny_detr(num_classes, num_queries)
    model.eval()

    images = torch.randn(B, 3, H, W)
    mask = torch.zeros(B, H, W, dtype=torch.bool)
    mask[1, :, W // 2:] = True
    with torch.no_grad():
        out = model(NestedTensor(images, mask))

    print(f"  입력 {tuple(images.shape)}")
    print(f"  pred_logits {tuple(out['pred_logits'].shape)}  = (B, num_queries, num_classes+1)")
    print(f"  pred_boxes  {tuple(out['pred_boxes'].shape)}  = (B, num_queries, 4)")

    assert out["pred_logits"].shape == (B, num_queries, num_classes + 1)
    assert out["pred_boxes"].shape == (B, num_queries, 4)
    # 마지막 클래스가 "no-object" 슬롯이다. 그래서 +1.
    assert (out["pred_boxes"] >= 0).all() and (out["pred_boxes"] <= 1).all(), \
        "박스는 sigmoid를 거쳐 [0, 1] 정규화 좌표로 나온다"
    print("  [ok] 박스는 sigmoid를 거쳐 항상 [0, 1] 범위")

    # 이미지 크기가 바뀌어도 예측 개수는 그대로 -- NMS도, 앵커도 없다.
    with torch.no_grad():
        out_big = model(NestedTensor(torch.randn(1, 3, 128, 160),
                                     torch.zeros(1, 128, 160, dtype=torch.bool)))
    assert out_big["pred_logits"].shape == (1, num_queries, num_classes + 1)
    print("  [ok] 이미지 크기가 달라져도 예측 개수는 항상 num_queries개")
    print("       -> 앵커도 NMS도 없이 '집합'을 바로 뱉는 것이 DETR의 핵심")

    # aux_loss를 켜면 디코더 중간 레이어 출력이 따라 나온다 (마지막 층 제외).
    aux_model = _build_tiny_detr(num_classes, num_queries, aux_loss=True)
    aux_model.eval()
    with torch.no_grad():
        aux_out = aux_model(NestedTensor(images, mask))
    n_dec = aux_model.transformer.decoder.num_layers
    print(f"  aux_loss=True -> 'aux_outputs' {len(aux_out['aux_outputs'])}개 "
          f"(디코더 {n_dec}층 중 마지막 제외)")
    assert len(aux_out["aux_outputs"]) == n_dec - 1
    assert "aux_outputs" not in out, "aux_loss=False면 키 자체가 없다"
    print("  [ok] aux_outputs는 레이어마다 같은 손실을 걸기 위한 중간 출력")

    print()
    print("=" * 66)
    print("SetCriterion -- 매칭된 쌍에만 손실을 건다")
    print("=" * 66)
    from .matcher import HungarianMatcher

    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    criterion = SetCriterion(num_classes, matcher,
                             weight_dict={"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2},
                             eos_coef=0.1, losses=["labels", "boxes", "cardinality"])
    criterion.eval()

    targets = [
        {"labels": torch.tensor([1, 3]),
         "boxes": torch.tensor([[0.25, 0.25, 0.2, 0.2], [0.70, 0.60, 0.3, 0.2]])},
        {"labels": torch.tensor([2]),
         "boxes": torch.tensor([[0.50, 0.50, 0.4, 0.4]])},
    ]

    # 일부러 "정답과 똑같은" 예측을 만들어 본다. 앞쪽 쿼리에 정답을 심고
    # 나머지 쿼리는 no-object(마지막 클래스)를 강하게 예측하게 한다.
    perfect_logits = torch.zeros(B, num_queries, num_classes + 1)
    perfect_logits[..., num_classes] = 20.0
    perfect_boxes = torch.full((B, num_queries, 4), 0.5)
    for b, t in enumerate(targets):
        n = len(t["labels"])
        perfect_boxes[b, :n] = t["boxes"]
        for q in range(n):
            perfect_logits[b, q, num_classes] = 0.0
            perfect_logits[b, q, t["labels"][q]] = 20.0

    losses = criterion({"pred_logits": perfect_logits, "pred_boxes": perfect_boxes}, targets)
    print("  정답과 동일한 예측을 넣었을 때:")
    for k in sorted(losses):
        print(f"      {k:18s} {losses[k].item():.6f}")

    assert losses["loss_bbox"].item() < 1e-6, "완벽한 박스면 L1 손실은 0이어야 한다"
    assert losses["loss_giou"].item() < 1e-6, "완벽한 박스면 GIoU 손실도 0이어야 한다"
    assert losses["cardinality_error"].item() == 0.0, "물체 개수도 정확히 맞아야 한다"
    assert losses["loss_ce"].item() < 0.01, "분류 손실도 거의 0이어야 한다"
    print("  [ok] 완벽한 예측 -> 박스/GIoU 손실 0, 개수 오차 0")

    # 박스를 어긋나게 하면 손실이 커진다.
    worse = perfect_boxes.clone()
    worse[0, 0, :2] += 0.3
    losses_worse = criterion({"pred_logits": perfect_logits, "pred_boxes": worse}, targets)
    assert losses_worse["loss_bbox"].item() > losses["loss_bbox"].item()
    print(f"  [ok] 박스를 0.3 어긋내면 loss_bbox {losses['loss_bbox'].item():.4f}"
          f" -> {losses_worse['loss_bbox'].item():.4f}")

    # weight_dict에 없는 키는 로깅용이다 (class_error, cardinality_error).
    assert set(criterion.weight_dict) <= set(losses)
    print(f"  [ok] weight_dict {sorted(criterion.weight_dict)} 만 실제로 역전파된다")
    print("       class_error / cardinality_error는 보기용 지표")

    print()
    print("=" * 66)
    print("PostProcess -- 정규화된 cxcywh를 원본 픽셀 xyxy로 되돌린다")
    print("=" * 66)
    post = PostProcess()
    one = {"pred_logits": torch.zeros(1, 1, num_classes + 1),
           "pred_boxes": torch.tensor([[[0.5, 0.5, 0.5, 0.5]]])}
    one["pred_logits"][0, 0, 2] = 10.0
    target_sizes = torch.tensor([[100, 200]])       # (height, width)
    results = post(one, target_sizes)
    box = results[0]["boxes"][0]
    print(f"  cxcywh (0.5, 0.5, 0.5, 0.5) + 원본 100x200 -> xyxy {box.tolist()}")
    assert torch.allclose(box, torch.tensor([50.0, 25.0, 150.0, 75.0])), \
        "정규화 좌표 x 이미지 크기"
    assert results[0]["labels"].item() == 2
    print("  [ok] no-object 클래스는 점수 계산에서 빠지고, 좌표는 픽셀 단위가 된다")

    print()
    print("=" * 66)
    print("MLP -- 박스 회귀에 쓰이는 3층 FFN")
    print("=" * 66)
    mlp = MLP(input_dim=8, hidden_dim=16, output_dim=4, num_layers=3)
    y = mlp(torch.randn(2, 5, 8))
    print(f"  (2, 5, 8) -> {tuple(y.shape)}, 선형층 {len(mlp.layers)}개")
    assert y.shape == (2, 5, 4)
    assert len(mlp.layers) == 3
    # 마지막 층에는 ReLU가 없다. 있다면 음수가 절대 안 나온다.
    assert (y < 0).any(), "마지막 층에 활성화가 없으므로 음수가 나올 수 있어야 한다"
    print("  [ok] 마지막 층에는 활성화가 없다 (그래서 음수도 나온다)")

    print()
    print("모든 검사 통과.")


if __name__ == "__main__":
    _demo()
