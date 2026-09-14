# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import argparse
from typing import Dict, List, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn, Tensor

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        assert cost_class >= 0 and cost_bbox >= 0 and cost_giou >= 0, \
            f"matching costs are weights and must be non-negative, got "\
            f"class={cost_class}, bbox={cost_bbox}, giou={cost_giou}"
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs: Dict[str, Tensor],
                targets: List[Dict[str, Tensor]]) -> List[Tuple[Tensor, Tensor]]:
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        assert "pred_logits" in outputs and "pred_boxes" in outputs, \
            f"outputs must carry 'pred_logits' and 'pred_boxes', got {sorted(outputs)}"
        assert outputs["pred_logits"].dim() == 3, \
            f"expected pred_logits (B, num_queries, C), got {tuple(outputs['pred_logits'].shape)}"
        assert outputs["pred_boxes"].shape[:2] == outputs["pred_logits"].shape[:2], \
            f"pred_boxes {tuple(outputs['pred_boxes'].shape)} does not line up with "\
            f"pred_logits {tuple(outputs['pred_logits'].shape)}"
        assert outputs["pred_boxes"].shape[-1] == 4, \
            f"boxes must be (cx, cy, w, h), got last dim {outputs['pred_boxes'].shape[-1]}"
        assert len(targets) == outputs["pred_logits"].shape[0], \
            f"{outputs['pred_logits'].shape[0]} predictions but {len(targets)} targets"
        # 분류 비용은 out_prob[:, labels]로 뽑으므로 레이블이 범위를 벗어나면 여기서 터진다.
        # DETR의 num_classes는 "max_obj_id + 1"이라 헷갈리기 쉽다 -- detr.py의 build() 주석 참고.
        num_logits = outputs["pred_logits"].shape[-1]
        for i, t in enumerate(targets):
            assert t["labels"].numel() == 0 or int(t["labels"].max()) < num_logits, \
                f"target {i} has label id {int(t['labels'].max())} but the classifier only "\
                f"emits {num_logits} logits (num_classes + 1); raise num_classes"

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        assert C.shape == (bs, num_queries, len(tgt_ids)), \
            f"cost matrix should be (B, num_queries, total_targets), got {tuple(C.shape)}"

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        out = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
               for i, j in indices]
        # 1:1 매칭이므로 짝의 개수는 min(num_queries, 그 이미지의 타깃 수)와 정확히 같다.
        for b, (src_idx, tgt_idx) in enumerate(out):
            assert len(src_idx) == len(tgt_idx) == min(num_queries, sizes[b]), \
                f"image {b}: matched {len(src_idx)} pairs, expected {min(num_queries, sizes[b])}"
            assert len(torch.unique(src_idx)) == len(src_idx), f"image {b}: a query was matched twice"
        return out


def build_matcher(args: argparse.Namespace) -> HungarianMatcher:
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)


def _demo() -> None:
    """헝가리안 매칭을 직접 굴려보는 예시. 실행: python -m models.matcher"""
    torch.manual_seed(0)

    num_queries, num_classes = 4, 5
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)

    # 정답 3개. 박스는 (cx, cy, w, h)이고 이미지 크기로 정규화되어 있다.
    tgt_boxes = torch.tensor([[0.20, 0.20, 0.10, 0.10],
                              [0.70, 0.70, 0.20, 0.20],
                              [0.50, 0.30, 0.30, 0.10]])
    tgt_labels = torch.tensor([1, 2, 3])

    # 예측을 일부러 "섞어서" 만든다: query0 -> target1, query2 -> target0, query3 -> target2.
    # query1은 아무것과도 안 맞는 엉뚱한 박스.
    pred_boxes = torch.stack([tgt_boxes[1], torch.tensor([0.95, 0.05, 0.02, 0.02]),
                              tgt_boxes[0], tgt_boxes[2]]).unsqueeze(0)
    pred_logits = torch.zeros(1, num_queries, num_classes + 1)
    for q, t in [(0, 1), (2, 0), (3, 2)]:
        pred_logits[0, q, tgt_labels[t]] = 10.0

    outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
    targets = [{"labels": tgt_labels, "boxes": tgt_boxes}]

    print("=" * 66)
    print("HungarianMatcher -- 예측과 정답을 1:1로 짝지어 준다")
    print("=" * 66)
    (src_idx, tgt_idx), = matcher(outputs, targets)
    print(f"  쿼리 {num_queries}개, 정답 {len(tgt_labels)}개")
    print(f"  매칭: query {src_idx.tolist()}  <->  target {tgt_idx.tolist()}")

    assert src_idx.tolist() == [0, 2, 3], "일부러 심어둔 대응을 찾아내야 한다"
    assert tgt_idx.tolist() == [1, 0, 2], "각 쿼리가 자기 짝을 찾아야 한다"
    assert 1 not in src_idx.tolist(), "엉뚱한 query1은 짝이 없어야 한다 (= no-object)"
    print("  [ok] 심어둔 대응 관계를 정확히 복원했다")
    print("       짝을 못 찾은 query1은 학습 때 'no-object'로 분류된다")

    # 짝의 개수는 min(쿼리 수, 정답 수)이고, 한 쿼리가 두 번 쓰이지 않는다.
    assert len(src_idx) == len(tgt_idx) == min(num_queries, len(tgt_labels))
    assert len(set(src_idx.tolist())) == len(src_idx)
    assert sorted(tgt_idx.tolist()) == list(range(len(tgt_labels))), "정답은 모두 한 번씩 쓰인다"
    print("  [ok] 1:1 -- 쿼리도 정답도 중복 없이 한 번씩만 쓰인다")

    # --- 정답 순서를 바꾸면 매칭도 그만큼 따라 바뀐다 ---
    perm = torch.tensor([2, 0, 1])
    permuted = [{"labels": tgt_labels[perm], "boxes": tgt_boxes[perm]}]
    (src_p, tgt_p), = matcher(outputs, permuted)
    # 원래 target t는 이제 위치 perm^-1[t]에 있다.
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(len(perm))
    assert src_p.tolist() == src_idx.tolist(), "쿼리 쪽 결과는 그대로여야 한다"
    assert tgt_p.tolist() == inverse[tgt_idx].tolist(), "정답 인덱스만 재배열되어야 한다"
    print("  [ok] 정답 순서를 섞어도 같은 쌍을 찾아낸다 (집합 예측이므로 순서는 무의미)")

    # --- 정답이 없는 이미지 ---
    print()
    empty = {"labels": torch.zeros(0, dtype=torch.int64), "boxes": torch.zeros(0, 4)}
    batched = {"pred_logits": pred_logits.repeat(2, 1, 1),
               "pred_boxes": pred_boxes.repeat(2, 1, 1)}
    indices = matcher(batched, [{"labels": tgt_labels, "boxes": tgt_boxes}, empty])
    print(f"  배치 2장(정답 3개 / 0개) -> 매칭 쌍 개수 {[len(i) for i, _ in indices]}")
    assert len(indices) == 2
    assert len(indices[1][0]) == 0, "정답이 없으면 짝도 없다 -- 모든 쿼리가 no-object"
    print("  [ok] 정답이 없는 이미지는 빈 매칭을 돌려준다")

    # --- 정답이 쿼리보다 많으면 일부는 버려진다 ---
    many = {"labels": torch.arange(6) % num_classes, "boxes": torch.rand(6, 4) * 0.4 + 0.3}
    (src_m, tgt_m), = matcher(outputs, [many])
    print(f"  쿼리 {num_queries}개 < 정답 6개 -> 매칭 {len(src_m)}개 (나머지 정답은 버려짐)")
    assert len(src_m) == num_queries, "min(쿼리, 정답) = 쿼리 수"
    print("  [ok] 쿼리 수가 검출 가능한 물체 수의 상한이다")

    # --- 형태가 안 맞으면 assert가 잡아준다 ---
    print()
    try:
        matcher({"pred_logits": pred_logits, "pred_boxes": pred_boxes[..., :3]}, targets)
    except AssertionError as e:
        print(f"  [ok] 좌표가 4개가 아니면 거부됨 -> {e}")
    else:
        raise RuntimeError("박스 좌표가 3개인데 통과해버렸다")

    print()
    print("모든 검사 통과.")


if __name__ == "__main__":
    _demo()
