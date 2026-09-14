# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import argparse
import math
from typing import Final, Optional

import torch
from torch import nn, Tensor

from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats: int = 64, temperature: int = 10000,
                 normalize: bool = False, scale: Optional[float] = None):
        super().__init__()
        # sin/cos를 짝지어 interleave하므로 홀수면 stack에서 크기가 안 맞는다.
        assert num_pos_feats > 0 and num_pos_feats % 2 == 0, \
            f"num_pos_feats must be a positive even number, got {num_pos_feats}"
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor) -> Tensor:
        """Args:
            tensor_list.tensors : (B, C, H, W)  -- 값은 안 쓰고 device/dtype/배치 크기만 참조
            tensor_list.mask    : (B, H, W)     bool, True = 패딩
        Returns:
            pos : (B, 2 * num_pos_feats, H, W)  -- y 인코딩과 x 인코딩을 채널로 이어붙인 것
        """
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert x.dim() == 4, f"expected (B, C, H, W), got {x.shape}"
        assert mask is not None, "sine positional encoding needs the padding mask"
        assert mask.dim() == 3, f"expected mask (B, H, W), got {mask.shape}"
        assert mask.shape[0] == x.shape[0], \
            f"batch mismatch: {x.shape[0]} feature maps vs {mask.shape[0]} masks"
        assert mask.shape[-2:] == x.shape[-2:], \
            f"mask {mask.shape} does not match feature plane {x.shape}"
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        assert pos.shape[1] == 2 * self.num_pos_feats, \
            f"expected {2 * self.num_pos_feats} channels, got {pos.shape[1]}"
        assert pos.shape[-2:] == x.shape[-2:], \
            f"pos {pos.shape} lost the feature plane {x.shape}"
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    # 학습형 임베딩 테이블의 행/열 개수. 피처맵이 이보다 크면 인덱싱이 터진다.
    MAX_SIDE: Final[int] = 50

    def __init__(self, num_pos_feats: int = 256):
        super().__init__()
        assert num_pos_feats > 0, f"num_pos_feats must be positive, got {num_pos_feats}"
        self.row_embed = nn.Embedding(self.MAX_SIDE, num_pos_feats)
        self.col_embed = nn.Embedding(self.MAX_SIDE, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor) -> Tensor:
        """Args:
            tensor_list.tensors : (B, C, H, W)
        Returns:
            pos : (B, 2 * num_pos_feats, H, W)  -- 행 임베딩과 열 임베딩을 채널로 이어붙인 것
        """
        x = tensor_list.tensors
        assert x.dim() == 4, f"expected (B, C, H, W), got {x.shape}"
        h, w = x.shape[-2:]
        assert h <= self.MAX_SIDE and w <= self.MAX_SIDE, \
            f"learned positional encoding has only {self.MAX_SIDE}x{self.MAX_SIDE} slots, "\
            f"got feature map {h}x{w}"
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        assert pos.shape[0] == x.shape[0], f"batch mismatch: {pos.shape} vs {x.shape}"
        assert pos.shape[-2:] == x.shape[-2:], f"pos {pos.shape} does not match {x.shape}"
        return pos


def build_position_encoding(args: argparse.Namespace) -> nn.Module:
    assert args.hidden_dim % 2 == 0, \
        f"hidden_dim must be even (it is split into x/y halves), got {args.hidden_dim}"
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding


def _demo() -> None:
    """위치 인코딩을 직접 굴려보는 예시. 실행: python -m models.position_encoding"""
    torch.manual_seed(0)

    B, C, H, W = 2, 8, 5, 7
    feats = torch.randn(B, C, H, W)
    # 0번 이미지는 패딩 없음, 1번 이미지는 오른쪽 2열이 패딩이라고 하자.
    mask = torch.zeros(B, H, W, dtype=torch.bool)
    mask[1, :, -2:] = True
    samples = NestedTensor(feats, mask)

    print("=" * 62)
    print("PositionEmbeddingSine -- 학습 파라미터가 없는 고정 인코딩")
    print("=" * 62)
    sine = PositionEmbeddingSine(num_pos_feats=16, normalize=True)
    pos = sine(samples)
    print(f"  feats {tuple(feats.shape)} + mask {tuple(mask.shape)}")
    print(f"  ->    {tuple(pos.shape)}  = (B, 2*num_pos_feats, H, W)")

    assert pos.shape == (B, 2 * 16, H, W)
    assert sum(p.numel() for p in sine.parameters()) == 0, "sine 인코딩에는 학습 파라미터가 없다"

    # 인코딩은 피처 "값"과 무관하다. 같은 마스크면 같은 결과.
    other = sine(NestedTensor(torch.randn(B, C, H, W), mask))
    assert torch.equal(pos, other), "위치 인코딩은 피처 값이 아니라 위치에만 의존해야 한다"
    print("  [ok] 피처 값이 달라져도 인코딩은 동일하다")

    # 좌표는 not_mask.cumsum()으로 만든다. 즉 "왼쪽에서 여기까지 유효 픽셀 몇 개냐"다.
    # 그래서 유효 영역(앞 5열) 안에서는 두 이미지의 raw 좌표가 1..5로 똑같다.
    raw = PositionEmbeddingSine(num_pos_feats=16, normalize=False)(samples)
    assert torch.equal(raw[0, :, :, :5], raw[1, :, :, :5]), \
        "normalize=False면 유효 영역의 좌표는 두 이미지가 같아야 한다"
    print("  [ok] normalize=False: 유효 영역(앞 5열)의 좌표가 두 이미지에서 동일")

    # normalize=True는 각자의 유효 폭으로 나눠 [0, 2pi]로 맞춘다. 0번은 7로, 1번은 5로
    # 나누므로 같은 5열이어도 값이 달라진다 -- 패딩이 좌표계를 바꾸지 않게 하는 장치.
    assert not torch.equal(pos[0, :, :, :5], pos[1, :, :, :5]), \
        "normalize=True면 유효 폭이 다른 이미지끼리 값이 달라야 한다"
    print("  [ok] normalize=True: 각 이미지의 유효 폭(7 vs 5)으로 나눠 스케일이 달라짐")

    print()
    print("=" * 62)
    print("PositionEmbeddingLearned -- 행/열 임베딩 테이블을 학습")
    print("=" * 62)
    learned = PositionEmbeddingLearned(num_pos_feats=16)
    pos_l = learned(samples)
    n_params = sum(p.numel() for p in learned.parameters())
    print(f"  ->    {tuple(pos_l.shape)}  = (B, 2*num_pos_feats, H, W)")
    print(f"  학습 파라미터 {n_params}개 = 행 {learned.MAX_SIDE}개 + 열 {learned.MAX_SIDE}개, 각 16차원")

    assert pos_l.shape == (B, 2 * 16, H, W)
    assert n_params == 2 * learned.MAX_SIDE * 16
    # 학습형은 마스크를 안 보므로 배치 안의 모든 이미지가 동일한 인코딩을 받는다.
    assert torch.equal(pos_l[0], pos_l[1]), "학습형 인코딩은 패딩을 고려하지 않는다"
    print("  [ok] 학습형은 마스크를 보지 않아 배치 전체가 같은 인코딩을 받는다")

    # 테이블이 50x50뿐이라 그보다 큰 피처맵은 assert로 막힌다.
    big = NestedTensor(torch.randn(1, C, 60, 60), torch.zeros(1, 60, 60, dtype=torch.bool))
    try:
        learned(big)
    except AssertionError as e:
        print(f"  [ok] 60x60 피처맵은 거부됨 -> {e}")
    else:
        raise RuntimeError("50을 넘는 피처맵인데 통과해버렸다")

    print()
    print("모든 검사 통과.")


if __name__ == "__main__":
    _demo()
