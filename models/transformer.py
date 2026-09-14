# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import argparse
import copy
from typing import Callable, Optional, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", normalize_before: bool = False,
                 return_intermediate_dec: bool = False):
        super().__init__()
        assert d_model % nhead == 0, \
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: Tensor, mask: Tensor, query_embed: Tensor,
                pos_embed: Tensor) -> Tuple[Tensor, Tensor]:
        """Args:
            src         : (B, d_model, H, W)   -- input_proj를 통과한 백본 피처
            mask        : (B, H, W)            bool, True = 패딩
            query_embed : (num_queries, d_model)
            pos_embed   : (B, d_model, H, W)   -- src와 같은 모양
        Returns:
            hs     : (num_decoder_layers, B, num_queries, d_model)
            memory : (B, d_model, H, W)        -- 인코더 출력을 다시 2D로 되돌린 것

        내부적으로는 (B, C, H, W)를 (H*W, B, C) 시퀀스로 펴서 쓴다. 이 구현은
        batch_first=False 규약이라 시퀀스 축이 맨 앞에 온다.
        """
        assert src.dim() == 4, f"expected src (B, C, H, W), got {src.shape}"
        assert src.shape[1] == self.d_model, \
            f"src has {src.shape[1]} channels but the transformer expects d_model={self.d_model}"
        assert pos_embed.shape == src.shape, \
            f"pos_embed {pos_embed.shape} must match src {src.shape}"
        assert mask.dim() == 3, f"expected mask (B, H, W), got {mask.shape}"
        assert mask.shape[0] == src.shape[0], \
            f"batch mismatch: src {src.shape[0]} vs mask {mask.shape[0]}"
        assert mask.shape[-2:] == src.shape[-2:], \
            f"mask {mask.shape} must match the feature plane of src {src.shape}"
        assert query_embed.dim() == 2, \
            f"expected query_embed (num_queries, d_model), got {query_embed.shape}"
        assert query_embed.shape[1] == self.d_model, \
            f"query_embed width {query_embed.shape[1]} != d_model {self.d_model}"

        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        assert memory.shape == src.shape, \
            f"encoder must preserve (S, B, C): {src.shape} -> {memory.shape}"
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer: nn.Module, num_layers: int,
                 norm: Optional[nn.Module] = None):
        assert num_layers > 0, f"num_layers must be positive, got {num_layers}"
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None) -> Tensor:
        """Args:
            src                  : (S, B, C)  -- S = H*W, 시퀀스 축이 앞
            src_key_padding_mask : (B, S)     bool, True = 무시할 위치
            pos                  : (S, B, C)  -- src와 같은 모양
        Returns:
            (S, B, C)
        """
        assert src.dim() == 3, f"encoder expects (S, B, C), got {src.shape}"
        if pos is not None:
            assert pos.shape == src.shape, f"pos {pos.shape} must match src {src.shape}"
        if src_key_padding_mask is not None:
            # key_padding_mask만 batch-first다: (B, S). src는 (S, B, C).
            assert src_key_padding_mask.dim() == 2, \
                f"key_padding_mask should be (B, S), got {src_key_padding_mask.shape}"
            assert src_key_padding_mask.shape[0] == src.shape[1], \
                f"batch mismatch: src has B={src.shape[1]}, mask has {src_key_padding_mask.shape[0]}"
            assert src_key_padding_mask.shape[1] == src.shape[0], \
                f"length mismatch: src has S={src.shape[0]}, mask has {src_key_padding_mask.shape[1]}"

        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer: nn.Module, num_layers: int,
                 norm: Optional[nn.Module] = None, return_intermediate: bool = False):
        assert num_layers > 0, f"num_layers must be positive, got {num_layers}"
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None) -> Tensor:
        """Args:
            tgt       : (Q, B, C)  -- Q = num_queries. 첫 레이어에는 0으로 들어온다.
            memory    : (S, B, C)  -- 인코더 출력
            pos       : (S, B, C)  -- memory(=이미지 토큰)용 위치 인코딩
            query_pos : (Q, B, C)  -- object query 임베딩
        Returns:
            return_intermediate이면 (num_layers, Q, B, C), 아니면 (1, Q, B, C).
            어느 쪽이든 앞에 레이어 축이 하나 붙어 나간다.
        """
        assert tgt.dim() == 3, f"decoder expects tgt (Q, B, C), got {tgt.shape}"
        assert memory.dim() == 3, f"decoder expects memory (S, B, C), got {memory.shape}"
        assert tgt.shape[1] == memory.shape[1], \
            f"batch mismatch: tgt B={tgt.shape[1]}, memory B={memory.shape[1]}"
        assert tgt.shape[2] == memory.shape[2], \
            f"width mismatch: tgt C={tgt.shape[2]}, memory C={memory.shape[2]}"
        if pos is not None:
            assert pos.shape == memory.shape, \
                f"pos {pos.shape} must match memory {memory.shape}"
        if query_pos is not None:
            assert query_pos.shape == tgt.shape, \
                f"query_pos {query_pos.shape} must match tgt {tgt.shape}"

        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            stacked = torch.stack(intermediate)
            assert stacked.shape[0] == self.num_layers, \
                f"expected one output per layer ({self.num_layers}), got {stacked.shape[0]}"
            return stacked

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu",
                 normalize_before: bool = False):
        assert d_model % nhead == 0, \
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        """위치 인코딩을 더해서 query/key를 만든다. value에는 더하지 않는다.

        DETR은 pos를 입력에 한 번만 더하는 게 아니라 매 어텐션마다 q/k에 더한다.
        브로드캐스팅으로 조용히 틀린 모양이 섞이지 않도록 형태를 못 박는다.
        """
        if pos is None:
            return tensor
        assert pos.shape == tensor.shape, \
            f"pos {pos.shape} must match the tensor it is added to {tensor.shape}"
        return tensor + pos

    def forward_post(self,
                     src: Tensor,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None) -> Tensor:
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src: Tensor,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None) -> Tensor:
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None) -> Tensor:
        """(S, B, C) -> (S, B, C). normalize_before면 pre-LN, 아니면 post-LN."""
        assert src.dim() == 3, f"encoder layer expects (S, B, C), got {src.shape}"
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu",
                 normalize_before: bool = False):
        assert d_model % nhead == 0, \
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        """위치 인코딩을 더해서 query/key를 만든다. value에는 더하지 않는다.

        DETR은 pos를 입력에 한 번만 더하는 게 아니라 매 어텐션마다 q/k에 더한다.
        브로드캐스팅으로 조용히 틀린 모양이 섞이지 않도록 형태를 못 박는다.
        """
        if pos is None:
            return tensor
        assert pos.shape == tensor.shape, \
            f"pos {pos.shape} must match the tensor it is added to {tensor.shape}"
        return tensor + pos

    def forward_post(self, tgt: Tensor, memory: Tensor,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None) -> Tensor:
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt: Tensor, memory: Tensor,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None) -> Tensor:
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None) -> Tensor:
        """(Q, B, C) -> (Q, B, C). self-attn(쿼리끼리) 다음 cross-attn(쿼리->이미지)."""
        assert tgt.dim() == 3, f"decoder layer expects tgt (Q, B, C), got {tgt.shape}"
        assert memory.dim() == 3, f"decoder layer expects memory (S, B, C), got {memory.shape}"
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args: argparse.Namespace) -> Transformer:
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _demo() -> None:
    """트랜스포머를 직접 굴려보는 예시. 실행: python -m models.transformer"""
    torch.manual_seed(0)

    B, d_model, H, W = 2, 32, 4, 5
    num_queries, nhead = 6, 4
    S = H * W

    model = Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=2,
                        num_decoder_layers=2, dim_feedforward=64,
                        return_intermediate_dec=True)
    model.eval()  # dropout을 꺼야 같은 입력에 같은 출력이 나온다

    src = torch.randn(B, d_model, H, W)          # input_proj를 통과한 백본 피처라고 치자
    pos_embed = torch.randn(B, d_model, H, W)    # 같은 모양의 위치 인코딩
    query_embed = torch.randn(num_queries, d_model)
    mask = torch.zeros(B, H, W, dtype=torch.bool)
    mask[1, :, -2:] = True                       # 1번 이미지의 오른쪽 2열은 패딩

    print("=" * 62)
    print("Transformer -- (B, C, H, W) 를 (H*W, B, C) 시퀀스로 펴서 처리")
    print("=" * 62)
    with torch.no_grad():
        hs, memory = model(src, mask, query_embed, pos_embed)
    print(f"  src         {tuple(src.shape)}   -> 시퀀스 길이 S = H*W = {S}")
    print(f"  query_embed {tuple(query_embed.shape)}")
    print(f"  hs          {tuple(hs.shape)}  = (num_decoder_layers, B, num_queries, d_model)")
    print(f"  memory      {tuple(memory.shape)}   = 인코더 출력을 다시 2D로")

    assert hs.shape == (2, B, num_queries, d_model)
    assert memory.shape == src.shape

    # --- 패딩 위치의 값은 결과에 영향을 주면 안 된다 ---
    # 유효 토큰은 key_padding_mask 때문에 패딩 토큰을 key로 보지 않고,
    # 디코더도 memory_key_padding_mask로 패딩을 가린다. 따라서 패딩 자리의
    # 입력값을 아무리 휘저어도 hs는 그대로여야 한다.
    src_perturbed = src.clone()
    src_perturbed[1, :, :, -2:] = torch.randn(d_model, H, 2) * 100
    with torch.no_grad():
        hs2, _ = model(src_perturbed, mask, query_embed, pos_embed)
    assert torch.equal(hs, hs2), "패딩 위치의 값이 디코더 출력에 새어 들어갔다"
    print("  [ok] 패딩 자리의 값을 100배로 흔들어도 hs는 완전히 동일")

    # --- 쿼리마다 다른 결과가 나와야 한다 ---
    # object query는 "서로 다른 것을 찾으라"고 학습되는 슬롯이다. 초기값이 같으면
    # 자기 어텐션을 거쳐도 서로 구별되지 않는다.
    same_queries = query_embed[:1].repeat(num_queries, 1)
    with torch.no_grad():
        hs_same, _ = model(src, mask, same_queries, pos_embed)
    assert torch.allclose(hs_same[-1, 0, 0], hs_same[-1, 0, 1], atol=1e-6), \
        "쿼리 임베딩이 같으면 출력도 같아야 한다 (대칭이 깨질 이유가 없다)"
    assert not torch.allclose(hs[-1, 0, 0], hs[-1, 0, 1], atol=1e-6), \
        "쿼리 임베딩이 다르면 출력도 달라야 한다"
    print("  [ok] 쿼리 임베딩이 같으면 출력도 같고, 다르면 달라진다")
    print("       -> object query가 서로 다른 슬롯이 되는 유일한 근거는 이 초기 임베딩이다")

    # --- 위치 인코딩이 없으면 인코더는 순서를 모른다 ---
    # pos를 0으로 주고 토큰 순서를 섞으면, 출력도 똑같이 섞인 것이어야 한다.
    print()
    print("=" * 62)
    print("TransformerEncoder -- 위치 인코딩이 없으면 순열 등변(permutation equivariant)")
    print("=" * 62)
    enc = model.encoder
    seq = torch.randn(S, 1, d_model)
    perm = torch.randperm(S)
    with torch.no_grad():
        out_plain = enc(seq, pos=None)
        out_perm = enc(seq[perm], pos=None)
    assert torch.allclose(out_plain[perm], out_perm, atol=1e-5), \
        "pos 없이는 토큰을 섞으면 출력도 똑같이 섞여야 한다"
    print("  [ok] pos=None이면 입력을 섞은 만큼 출력도 그대로 섞인다")

    pos_seq = torch.randn(S, 1, d_model)
    with torch.no_grad():
        out_pos = enc(seq, pos=pos_seq)
        out_pos_perm = enc(seq[perm], pos=pos_seq[perm])
    assert torch.allclose(out_pos[perm], out_pos_perm, atol=1e-5), \
        "pos를 같이 섞으면 결과도 같이 섞여야 한다"
    with torch.no_grad():
        out_pos_only_seq = enc(seq[perm], pos=pos_seq)
    assert not torch.allclose(out_pos[perm], out_pos_only_seq, atol=1e-5), \
        "토큰만 섞고 pos를 그대로 두면 결과가 달라져야 한다"
    print("  [ok] pos를 주면 순서가 의미를 갖는다 (토큰만 섞으면 결과가 달라짐)")

    # --- 디코더는 레이어마다 한 장씩 쌓아 내보낸다 (aux loss용) ---
    print()
    print("=" * 62)
    print("TransformerDecoder -- return_intermediate로 레이어별 출력을 모두 반환")
    print("=" * 62)
    tgt = torch.zeros(num_queries, 1, d_model)
    mem = torch.randn(S, 1, d_model)
    with torch.no_grad():
        stacked = model.decoder(tgt, mem, query_pos=query_embed.unsqueeze(1))
    print(f"  return_intermediate=True -> {tuple(stacked.shape)}  (레이어 축이 맨 앞)")
    assert stacked.shape == (2, num_queries, 1, d_model)

    shallow = TransformerDecoder(
        TransformerDecoderLayer(d_model, nhead, dim_feedforward=64),
        num_layers=2, norm=nn.LayerNorm(d_model), return_intermediate=False)
    shallow.eval()
    with torch.no_grad():
        last_only = shallow(tgt, mem, query_pos=query_embed.unsqueeze(1))
    print(f"  return_intermediate=False -> {tuple(last_only.shape)}  (축은 있지만 길이 1)")
    assert last_only.shape == (1, num_queries, 1, d_model)
    print("  [ok] 어느 쪽이든 레이어 축이 하나 붙어 나가서 부르는 쪽 코드가 같아진다")

    # --- 형태가 안 맞으면 assert가 잡아준다 ---
    print()
    try:
        model(src, mask, query_embed[:, :8], pos_embed)
    except AssertionError as e:
        print(f"  [ok] 잘못된 query_embed 폭은 거부됨 -> {e}")
    else:
        raise RuntimeError("d_model이 안 맞는데 통과해버렸다")

    print()
    print("모든 검사 통과.")


if __name__ == "__main__":
    _demo()
