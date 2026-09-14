# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
import argparse
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor
from torchvision.models._utils import IntermediateLayerGetter
from typing import Any, Dict, List, Tuple

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n: int):
        assert n > 0, f"channel count must be positive, got {n}"
        super(FrozenBatchNorm2d, self).__init__()
        # register_buffer: 모듈에 "학습되지 않는 텐서"를 등록한다.
        # nn.Parameter처럼 state_dict에 저장되고 .to(device)로 같이 옮겨지지만,
        # requires_grad=False라 optimizer가 갱신하지 않는다 -> BN 통계/affine을 고정하는 용도.
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict: Dict[str, Tensor], prefix: str,
                              local_metadata: Dict[str, Any], strict: bool,
                              missing_keys: List[str], unexpected_keys: List[str],
                              error_msgs: List[str]) -> None:
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: Tensor) -> Tensor:
        """추론용 BN을 채널별 affine 변환 하나로 접어서 적용한다.

        BN 추론식을 정리하면 채널마다 상수 두 개(scale, bias)만 남는다:
            scale = weight / sqrt(running_var + eps)
            bias  = bias - running_mean * scale
            y     = x * scale + bias

        Shapes:
            x                : (N, C, H, W)
            버퍼(weight 등)   : (C,)  -- BN 파라미터는 "채널마다 하나"다.
            reshape 후        : (1, C, 1, 1)

        reshape이 필요한 이유: 브로드캐스팅은 뒤쪽 축부터 정렬하므로 (C,)를
        그대로 쓰면 채널이 아니라 W 축에 곱해진다. (1, C, 1, 1)로 만들면
        1인 축(N, H, W)은 브로드캐스트되고 C만 실제로 매칭된다.

        reshape을 전부 앞으로 몰아둔 건 마지막 줄을 순수 element-wise 체인으로
        만들기 위해서다. JIT 퓨저가 x * scale + bias를 커널 하나로 합칠 수 있다
        (= fuser-friendly). 무거운 N*C*H*W 연산은 마지막 한 줄뿐이고, scale/bias
        계산은 C개 원소로 끝난다.
        """
        assert x.dim() == 4, f"FrozenBatchNorm2d expects (N, C, H, W), got {x.shape}"
        assert x.shape[1] == self.weight.numel(), \
            f"channel mismatch: input has C={x.shape[1]}, this BN was built for {self.weight.numel()}"

        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        assert num_channels > 0, f"num_channels must be positive, got {num_channels}"
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor) -> Dict[str, NestedTensor]:
        """백본 피처를 뽑고, 패딩 마스크를 피처맵 해상도에 맞춰 내려준다.

        배치를 만들 때 크기가 다른 이미지들을 최대 크기에 맞춰 0으로 패딩하므로
        (util.misc.nested_tensor_from_tensor_list 참고), 어디가 진짜 픽셀이고
        어디가 패딩인지 알려주는 mask가 따라다닌다. True = 패딩.
        피처맵은 stride 때문에 입력보다 작아지니 mask도 같이 줄여야 한다.

        Shapes:
            tensor_list.tensors : (B, 3, H, W)
            tensor_list.mask    : (B, H, W)          bool, True = 패딩
            x (피처맵)          : (B, C, H/s, W/s)   s = 32 (dilation이면 16)
            mask (출력)         : (B, H/s, W/s)      bool

        interpolate 한 줄이 복잡해 보이는 이유는 타입/차원을 억지로 맞추기 때문:
            .unsqueeze(0)    -> (1, B, H, W). interpolate는 4D (N, C, H, W)를
                                요구하는데 mask는 3D다. 앞에 더미 축을 붙여
                                배치 B를 "채널"인 척 취급시킨다. 채널별로
                                독립 리사이즈되므로 결과는 동일하다.
                                (원본 DETR 코드는 같은 뜻으로 m[None]을 썼다.)
            .float()         -> interpolate는 bool을 못 받는다.
            size=x.shape[-2:] -> 피처맵의 (H/s, W/s)에 정확히 맞춘다. stride를
                                직접 계산하지 않고 실제 출력 크기를 그대로 쓴다.
            (mode 기본값은 'nearest'라 값이 섞이지 않고 0.0/1.0만 남는다.
             즉 저해상도 셀은 대응되는 원본 픽셀이 패딩이면 패딩이 된다.)
            .to(torch.bool)  -> 다시 마스크로.
            .squeeze(0)      -> 붙였던 더미 축 제거 -> (B, H/s, W/s).

        이 마스크는 나중에 트랜스포머의 key_padding_mask로 쓰여서 패딩 위치에
        어텐션이 가지 않게 막는다.
        """
        images = tensor_list.tensors
        m = tensor_list.mask
        assert images.dim() == 4, f"expected images (B, 3, H, W), got {images.shape}"
        assert m is not None, "backbone needs the padding mask; build the batch with NestedTensor"
        assert m.dim() == 3, f"expected mask (B, H, W), got {m.shape}"
        assert m.shape[0] == images.shape[0], \
            f"batch mismatch: {images.shape[0]} images vs {m.shape[0]} masks"
        assert m.shape[-2:] == images.shape[-2:], \
            f"mask must cover the image plane: mask {m.shape} vs images {images.shape}"

        xs = self.body(images)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            assert x.dim() == 4, f"feature map '{name}' should be (B, C, H, W), got {x.shape}"
            mask = F.interpolate(m.unsqueeze(0).float(), size=x.shape[-2:]).to(torch.bool).squeeze(0)
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        assert hasattr(torchvision.models, name), f"torchvision has no model named {name!r}"
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    """백본과 위치 인코딩을 묶어, 피처맵과 그에 대응하는 pos embedding을 함께 내보낸다."""

    def __init__(self, backbone: nn.Module, position_embedding: nn.Module):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor) -> Tuple[List[NestedTensor], List[Tensor]]:
        """Returns:
            out: 피처맵 리스트. 각 원소는 (B, C_l, H_l, W_l) 텐서 + 같은 해상도의 mask.
            pos: 위치 인코딩 리스트. out[i]와 해상도가 1:1로 대응하고 채널은 hidden_dim.
        """
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            p = self[1](x).to(x.tensors.dtype)
            assert p.dim() == 4, f"position encoding should be (B, C, H, W), got {p.shape}"
            assert p.shape[-2:] == x.tensors.shape[-2:], \
                f"pos embedding {p.shape} does not match feature map {x.tensors.shape}"
            pos.append(p)

        assert len(out) == len(pos), f"{len(out)} feature maps but {len(pos)} pos embeddings"
        return out, pos


def build_backbone(args: argparse.Namespace) -> Joiner:
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


def _demo() -> None:
    """백본을 직접 굴려보는 예시. 실행: python -m models.backbone

    사전학습 가중치를 내려받지 않도록 resnet18을 weights=None으로 직접 만든다.
    """
    torch.manual_seed(0)

    print("=" * 66)
    print("FrozenBatchNorm2d -- 학습되지 않는 BN")
    print("=" * 66)
    C = 8
    fbn = FrozenBatchNorm2d(C)
    # 그럴듯한 통계값을 심어준다 (보통은 사전학습 체크포인트에서 로드된다).
    fbn.weight.copy_(torch.rand(C) + 0.5)
    fbn.bias.copy_(torch.randn(C))
    fbn.running_mean.copy_(torch.randn(C))
    fbn.running_var.copy_(torch.rand(C) + 0.5)

    print(f"  학습 파라미터: {sum(p.numel() for p in fbn.parameters())}개")
    print(f"  버퍼: {[n for n, _ in fbn.named_buffers()]}")
    assert len(list(fbn.parameters())) == 0, "FrozenBatchNorm2d에는 nn.Parameter가 하나도 없다"
    assert set(dict(fbn.named_buffers())) == {"weight", "bias", "running_mean", "running_var"}
    # 버퍼는 state_dict에 그대로 실려서 체크포인트로 오간다.
    assert set(fbn.state_dict()) == set(dict(fbn.named_buffers()))
    print("  [ok] 파라미터 0개, 버퍼 4개, state_dict에는 버퍼가 그대로 들어간다")

    # eval 모드의 nn.BatchNorm2d와 수치가 같아야 한다 -- 같은 식을 접어놓은 것이므로.
    ref = nn.BatchNorm2d(C, eps=1e-5)
    ref.weight.data.copy_(fbn.weight)
    ref.bias.data.copy_(fbn.bias)
    ref.running_mean.data.copy_(fbn.running_mean)
    ref.running_var.data.copy_(fbn.running_var)
    ref.eval()
    x = torch.randn(2, C, 5, 7)
    assert torch.allclose(fbn(x), ref(x), atol=1e-6), "eval 모드 BN과 값이 달라졌다"
    print("  [ok] eval 모드 nn.BatchNorm2d와 수치가 일치")

    # train() 을 불러도 통계가 갱신되지 않는다 -- 진짜 BN이라면 바뀐다.
    fbn.train()
    before = fbn.running_mean.clone()
    fbn(torch.randn(64, C, 5, 7) * 10 + 5)
    assert torch.equal(fbn.running_mean, before), "frozen인데 통계가 움직였다"
    ref.train()
    ref(torch.randn(64, C, 5, 7) * 10 + 5)
    assert not torch.equal(ref.running_mean, before), "대조군인 BN은 통계가 움직여야 한다"
    print("  [ok] train() 중에도 통계가 고정 (일반 BN은 움직인다)")

    try:
        fbn(torch.randn(2, C + 1, 5, 7))
    except AssertionError as e:
        print(f"  [ok] 채널 수가 안 맞으면 거부됨 -> {e}")
    else:
        raise RuntimeError("채널이 안 맞는데 통과해버렸다")

    print()
    print("=" * 66)
    print("BackboneBase -- 피처맵과 함께 마스크를 해상도에 맞춰 내려준다")
    print("=" * 66)
    resnet = torchvision.models.resnet18(weights=None, norm_layer=FrozenBatchNorm2d)
    backbone = BackboneBase(resnet, train_backbone=False, num_channels=512,
                            return_interm_layers=False)
    backbone.eval()

    # 폭은 stride 32의 짝수 배로 잡는다. 마스크는 'nearest'로 줄이기 때문에
    # 피처맵이 너무 작으면 패딩 경계가 격자에 스냅되어 비율이 어긋난다.
    B, H, W = 2, 64, 128
    images = torch.randn(B, 3, H, W)
    mask = torch.zeros(B, H, W, dtype=torch.bool)
    mask[1, :, W // 2:] = True          # 1번 이미지는 오른쪽 절반이 패딩
    with torch.no_grad():
        feats = backbone(NestedTensor(images, mask))

    name, nt = next(iter(feats.items()))
    fh, fw = nt.tensors.shape[-2:]
    print(f"  입력 {tuple(images.shape)}, 마스크 {tuple(mask.shape)}")
    print(f"  출력 '{name}': 피처 {tuple(nt.tensors.shape)}, 마스크 {tuple(nt.mask.shape)}")
    print(f"  stride = {H // fh} (H {H} -> {fh}), {W // fw} (W {W} -> {fw})")

    assert nt.tensors.shape == (B, 512, H // 32, W // 32), "resnet의 총 stride는 32다"
    assert nt.mask.shape == (B, fh, fw), "마스크가 피처맵 해상도로 따라와야 한다"
    assert not nt.mask[0].any(), "0번 이미지는 패딩이 없다"
    # 오른쪽 절반이 패딩이었으므로 줄어든 마스크도 오른쪽 절반이 True여야 한다.
    assert not nt.mask[1, :, :fw // 2].any(), "왼쪽 절반은 유효 픽셀이어야 한다"
    assert nt.mask[1, :, fw // 2:].all(), "오른쪽 절반은 패딩으로 남아야 한다"
    assert nt.mask[1].float().mean().item() == 0.5
    print(f"  [ok] 마스크의 패딩 비율이 보존됨: {nt.mask[1].float().mean():.2f} (원본 0.50)")

    assert all(not p.requires_grad for p in backbone.parameters()), \
        "train_backbone=False면 모든 파라미터가 얼어야 한다"
    print("  [ok] train_backbone=False -> 모든 파라미터 requires_grad=False")

    # return_interm_layers=True면 네 단계를 모두 돌려준다 (파놉틱 마스크 헤드용).
    multi = BackboneBase(torchvision.models.resnet18(weights=None, norm_layer=FrozenBatchNorm2d),
                         train_backbone=True, num_channels=512, return_interm_layers=True)
    multi.eval()
    with torch.no_grad():
        stages = multi(NestedTensor(images, mask))
    shapes = {k: tuple(v.tensors.shape) for k, v in stages.items()}
    print(f"  return_interm_layers=True -> {len(stages)}개 단계")
    for k, v in shapes.items():
        print(f"      '{k}': {v}")
    assert len(stages) == 4
    assert [v[1] for v in shapes.values()] == [64, 128, 256, 512], "resnet18의 단계별 채널 수"

    print()
    print("=" * 66)
    print("Joiner -- 백본 + 위치 인코딩을 한 모듈로 묶는다")
    print("=" * 66)
    from .position_encoding import PositionEmbeddingSine

    joiner = Joiner(backbone, PositionEmbeddingSine(num_pos_feats=128, normalize=True))
    joiner.eval()
    with torch.no_grad():
        out, pos = joiner(NestedTensor(images, mask))
    print(f"  피처 {len(out)}개, 위치 인코딩 {len(pos)}개")
    print(f"      feature {tuple(out[0].tensors.shape)}  <->  pos {tuple(pos[0].shape)}")
    assert len(out) == len(pos) == 1
    assert pos[0].shape[-2:] == out[0].tensors.shape[-2:], "해상도가 1:1로 대응해야 한다"
    assert pos[0].shape[1] == 256, "2 * num_pos_feats = hidden_dim"
    print("  [ok] 피처맵과 위치 인코딩의 해상도가 정확히 대응")

    print()
    print("모든 검사 통과.")


if __name__ == "__main__":
    _demo()
