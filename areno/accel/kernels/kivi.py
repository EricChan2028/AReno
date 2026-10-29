"""KIVI low-bit KV-cache kernels (asymmetric 2/4-bit storage).

KIVI (Liu et al., 2024, https://arxiv.org/abs/2402.02750) stores keys with
per-channel quantization (one scale/zero per channel for a group of ``GROUP``
consecutive tokens) and values with per-token quantization (one scale/zero per
token for a group of ``GROUP`` consecutive channels). Scales and zeros are
fp16; codes are unsigned ``BITS``-bit integers produced by round-to-nearest
asymmetric min/max quantization.

The paged runtime needs two device operations:

* ``store_groups`` quantizes complete ``GROUP``-token groups of K/V and writes
  packed codes plus scale/zero metadata into paged cache blocks.
* ``decode_attention`` runs split-K single-token decode attention. Each program
  walks ``GROUP``-token tiles of one sequence, dequantizes the paged prefix in
  registers (no full-cache materialization), reads the recent full-precision
  residual tokens from the per-slot ring, and a second kernel merges splits.

Packing layout: ``8 // BITS`` codes share one byte. Byte ``j`` of a token row
holds channels ``j + i * (HEAD_DIM * BITS // 8)`` at bit offset ``i * BITS``, so
a broadcast shift followed by a row-major reshape restores channel order.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

# Largest finite fp16 value; scales/zeros saturate here instead of overflowing.
_FP16_MAX = 65504.0


@triton.jit
def _quantize(x, zero, scale, QMAX: tl.constexpr):
    # Round-to-nearest asymmetric quantization against the stored fp16
    # scale/zero, so dequantization reproduces exactly what was rounded. A zero
    # scale (constant group) maps every element to code 0, i.e. back to `zero`.
    inv = tl.where(scale > 0, 1.0 / scale, 0.0)
    codes = tl.floor((x - zero) * inv + 0.5)
    return tl.minimum(tl.maximum(codes, 0.0), QMAX).to(tl.int32)


@triton.jit
def _pack(codes, ROWS: tl.constexpr, HEAD_DIM: tl.constexpr, BITS: tl.constexpr):
    # (ROWS, HEAD_DIM) int codes -> (ROWS, HEAD_DIM * BITS // 8) uint8.
    # Channel c = i * packed_dim + j lands in byte j at bit offset i * BITS;
    # the fields are disjoint, so summing the shifted codes is a bitwise OR.
    PACK: tl.constexpr = 8 // BITS
    shifts = tl.arange(0, PACK) * BITS
    grouped = tl.reshape(codes, (ROWS, PACK, HEAD_DIM // PACK))
    return tl.sum(grouped << shifts[None, :, None], axis=1).to(tl.uint8)


@triton.jit
def _unpack(packed, ROWS: tl.constexpr, HEAD_DIM: tl.constexpr, BITS: tl.constexpr):
    # Inverse of `_pack`: (ROWS, packed_dim) uint8 -> (ROWS, HEAD_DIM) fp32 codes.
    PACK: tl.constexpr = 8 // BITS
    shifts = tl.arange(0, PACK) * BITS
    fields = (packed.to(tl.int32)[:, None, :] >> shifts[None, :, None]) & ((1 << BITS) - 1)
    return tl.reshape(fields, (ROWS, HEAD_DIM)).to(tl.float32)


@triton.jit
def _kivi_store_groups_kernel(
    K_SRC,
    V_SRC,
    SRC_ROW,
    SRC_POS,
    DST_BLOCK,
    DST_OFFSET,
    VALID,
    K_CODES,
    K_SCALE,
    K_ZERO,
    V_CODES,
    V_SCALE,
    V_ZERO,
    src_mod,
    stride_src_row,
    stride_src_head,
    stride_src_pos,
    stride_kc_block,
    stride_kc_head,
    stride_kc_tok,
    stride_ks_block,
    stride_ks_head,
    stride_ks_group,
    stride_vc_block,
    stride_vc_head,
    stride_vc_tok,
    stride_vs_block,
    stride_vs_head,
    stride_vs_tok,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BITS: tl.constexpr,
    HAS_VALID: tl.constexpr,
):
    # One program quantizes one GROUP-token group of one KV head. Grid:
    # program_id(0) = group, program_id(1) = KV head. Source token t of the
    # group is read at position (SRC_POS + t) % src_mod of row SRC_ROW, which
    # covers both flat prefill staging (src_mod larger than any position) and
    # the per-slot residual ring (src_mod = ring size).
    group_id = tl.program_id(0)
    head = tl.program_id(1)
    if HAS_VALID:
        # Decode-time transitions launch one candidate group per row; rows that
        # do not cross a group boundary this step exit without writing.
        if tl.load(VALID + group_id) == 0:
            return
    src_row = tl.load(SRC_ROW + group_id).to(tl.int64)
    src_pos = tl.load(SRC_POS + group_id).to(tl.int64)
    block = tl.load(DST_BLOCK + group_id).to(tl.int64)
    offset = tl.load(DST_OFFSET + group_id).to(tl.int64)
    QMAX: tl.constexpr = (1 << BITS) - 1
    tok = tl.arange(0, GROUP)
    dim = tl.arange(0, HEAD_DIM)
    pos = (src_pos + tok) % src_mod
    src = src_row * stride_src_row + head * stride_src_head + pos[:, None] * stride_src_pos + dim[None, :]
    k = tl.load(K_SRC + src).to(tl.float32)
    v = tl.load(V_SRC + src).to(tl.float32)

    # Keys: per-channel range over the group's tokens.
    k_min = tl.min(k, axis=0)
    k_scale = tl.minimum((tl.max(k, axis=0) - k_min) / QMAX, _FP16_MAX).to(tl.float16)
    k_zero = tl.minimum(tl.maximum(k_min, -_FP16_MAX), _FP16_MAX).to(tl.float16)
    k_codes = _quantize(k, k_zero.to(tl.float32)[None, :], k_scale.to(tl.float32)[None, :], QMAX)

    # Values: per-token range over each GROUP-channel slice of the head.
    v_grouped = tl.reshape(v, (GROUP, HEAD_DIM // GROUP, GROUP))
    v_min = tl.min(v_grouped, axis=2)
    v_scale = tl.minimum((tl.max(v_grouped, axis=2) - v_min) / QMAX, _FP16_MAX).to(tl.float16)
    v_zero = tl.minimum(tl.maximum(v_min, -_FP16_MAX), _FP16_MAX).to(tl.float16)
    v_codes = tl.reshape(
        _quantize(v_grouped, v_zero.to(tl.float32)[:, :, None], v_scale.to(tl.float32)[:, :, None], QMAX),
        (GROUP, HEAD_DIM),
    )

    packed = tl.arange(0, HEAD_DIM * BITS // 8)
    rows = (offset + tok)[:, None]
    kc = block * stride_kc_block + head * stride_kc_head + rows * stride_kc_tok + packed[None, :]
    tl.store(K_CODES + kc, _pack(k_codes, GROUP, HEAD_DIM, BITS))
    ks = block * stride_ks_block + head * stride_ks_head + (offset // GROUP) * stride_ks_group + dim
    tl.store(K_SCALE + ks, k_scale)
    tl.store(K_ZERO + ks, k_zero)
    vc = block * stride_vc_block + head * stride_vc_head + rows * stride_vc_tok + packed[None, :]
    tl.store(V_CODES + vc, _pack(v_codes, GROUP, HEAD_DIM, BITS))
    vgroup = tl.arange(0, HEAD_DIM // GROUP)
    vs = block * stride_vs_block + head * stride_vs_head + rows * stride_vs_tok + vgroup[None, :]
    tl.store(V_SCALE + vs, v_scale)
    tl.store(V_ZERO + vs, v_zero)


@triton.jit
def _kivi_decode_partial_kernel(
    Q,
    K_CODES,
    K_SCALE,
    K_ZERO,
    V_CODES,
    V_SCALE,
    V_ZERO,
    K_RING,
    V_RING,
    BLOCK_TABLE,
    CACHE_SEQLENS,
    SLOTS,
    PART_ACC,
    PART_MAX,
    PART_SUM,
    qk_scale,
    residual,
    ring_size,
    stride_q_row,
    stride_q_head,
    stride_kc_block,
    stride_kc_head,
    stride_kc_tok,
    stride_ks_block,
    stride_ks_head,
    stride_ks_group,
    stride_vc_block,
    stride_vc_head,
    stride_vc_tok,
    stride_vs_block,
    stride_vs_head,
    stride_vs_tok,
    stride_ring_slot,
    stride_ring_head,
    stride_ring_pos,
    stride_bt_row,
    stride_pa_row,
    stride_pa_head,
    stride_pa_split,
    stride_pm_row,
    stride_pm_head,
    QUERIES_PER_KV: tl.constexpr,
    HEADS_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP: tl.constexpr,
    BITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    # Grid: (row, KV head, split). The program serves all QUERIES_PER_KV query
    # heads that share this KV head (GQA), padded to HEADS_PAD rows for tl.dot.
    # CACHE_SEQLENS holds the tokens cached before this step; the new token at
    # that position has already been written to the ring, so the row attends
    # to `total = cached + 1` positions: the paged quantized prefix
    # [0, quantized) and the ring [quantized, total).
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    total = tl.load(CACHE_SEQLENS + row) + 1
    slot = tl.load(SLOTS + row).to(tl.int64)
    past = total - 1 - residual
    quantized = tl.where(past >= 0, (past // GROUP) * GROUP, 0)
    num_tiles = tl.cdiv(total, GROUP)
    per_split = tl.cdiv(num_tiles, NUM_SPLITS)
    tile_begin = split * per_split
    tile_end = tl.minimum(tile_begin + per_split, num_tiles)

    heads = tl.arange(0, HEADS_PAD)
    head_mask = heads < QUERIES_PER_KV
    q_heads = kv_head * QUERIES_PER_KV + heads
    dim = tl.arange(0, HEAD_DIM)
    tok = tl.arange(0, GROUP)
    packed = tl.arange(0, HEAD_DIM * BITS // 8)
    q = tl.load(
        Q + row * stride_q_row + q_heads[:, None] * stride_q_head + dim[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )
    m_i = tl.full([HEADS_PAD], float("-inf"), tl.float32)
    l_i = tl.zeros([HEADS_PAD], tl.float32)
    acc = tl.zeros([HEADS_PAD, HEAD_DIM], tl.float32)
    for tile in range(tile_begin, tile_end):
        start = tile * GROUP
        positions = start + tok
        valid = positions < total
        if start < quantized:
            # Quantized tiles are whole key groups inside one paged block
            # (GROUP divides the block size), so one block lookup serves them.
            block = tl.load(BLOCK_TABLE + row * stride_bt_row + start // BLOCK_SIZE).to(tl.int64)
            rows = (start % BLOCK_SIZE + tok)[:, None]
            kc = block * stride_kc_block + kv_head * stride_kc_head + rows * stride_kc_tok + packed[None, :]
            ks = block * stride_ks_block + kv_head * stride_ks_head + ((start % BLOCK_SIZE) // GROUP) * stride_ks_group
            k = _unpack(tl.load(K_CODES + kc), GROUP, HEAD_DIM, BITS)
            k = k * tl.load(K_SCALE + ks + dim).to(tl.float32)[None, :] + tl.load(K_ZERO + ks + dim).to(tl.float32)[None, :]
            vc = block * stride_vc_block + kv_head * stride_vc_head + rows * stride_vc_tok + packed[None, :]
            vs = block * stride_vs_block + kv_head * stride_vs_head + rows * stride_vs_tok + (dim // GROUP)[None, :]
            v = _unpack(tl.load(V_CODES + vc), GROUP, HEAD_DIM, BITS)
            v = v * tl.load(V_SCALE + vs).to(tl.float32) + tl.load(V_ZERO + vs).to(tl.float32)
        else:
            # Residual tokens live in the slot's ring at position % ring_size.
            ring = (
                slot * stride_ring_slot
                + kv_head * stride_ring_head
                + (positions % ring_size)[:, None] * stride_ring_pos
                + dim[None, :]
            )
            k = tl.load(K_RING + ring, mask=valid[:, None], other=0.0).to(tl.float32)
            v = tl.load(V_RING + ring, mask=valid[:, None], other=0.0).to(tl.float32)
        # Online softmax in base 2 (qk_scale already folds in log2(e)).
        scores = tl.dot(q, tl.trans(k.to(q.dtype))) * qk_scale
        scores = tl.where(valid[None, :], scores, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v.to(q.dtype))
        m_i = m_new

    pa = row * stride_pa_row + q_heads[:, None] * stride_pa_head + split * stride_pa_split + dim[None, :]
    tl.store(PART_ACC + pa, acc, mask=head_mask[:, None])
    pm = row * stride_pm_row + q_heads * stride_pm_head + split
    tl.store(PART_MAX + pm, m_i, mask=head_mask)
    tl.store(PART_SUM + pm, l_i, mask=head_mask)


@triton.jit
def _kivi_combine_kernel(
    PART_ACC,
    PART_MAX,
    PART_SUM,
    OUT,
    LSE,
    stride_pa_row,
    stride_pa_head,
    stride_pa_split,
    stride_pm_row,
    stride_pm_head,
    stride_o_row,
    stride_o_head,
    stride_lse_row,
    HEAD_DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SPLITS_PAD: tl.constexpr,
    HAS_LSE: tl.constexpr,
):
    # Grid: (row, query head). Merge the split partials with the usual
    # log-sum-exp rescaling. Empty splits carry (max=-inf, sum=0) and are
    # masked out so their uninitialized accumulators are never read.
    row = tl.program_id(0)
    head = tl.program_id(1)
    split = tl.arange(0, SPLITS_PAD)
    pm = row * stride_pm_row + head * stride_pm_head + split
    in_range = split < NUM_SPLITS
    m = tl.load(PART_MAX + pm, mask=in_range, other=float("-inf"))
    l = tl.load(PART_SUM + pm, mask=in_range, other=0.0)
    live = in_range & (l > 0)
    m_max = tl.max(tl.where(live, m, float("-inf")), axis=0)
    weight = tl.where(live, tl.exp2(m - m_max), 0.0)
    norm = tl.sum(l * weight, axis=0)
    dim = tl.arange(0, HEAD_DIM)
    pa = row * stride_pa_row + head * stride_pa_head + split[:, None] * stride_pa_split + dim[None, :]
    acc = tl.load(PART_ACC + pa, mask=live[:, None], other=0.0)
    out = tl.sum(acc * weight[:, None], axis=0) / norm
    tl.store(OUT + row * stride_o_row + head * stride_o_head + dim, out)
    if HAS_LSE:
        # Natural-log LSE so callers can merge with flash-attn's softmax_lse.
        tl.store(LSE + row * stride_lse_row + head, (m_max + tl.log2(norm)) * 0.6931471805599453)


def store_groups(
    k_src: torch.Tensor,
    v_src: torch.Tensor,
    src_strides: tuple[int, int, int],
    src_mod: int,
    src_row: torch.Tensor,
    src_pos: torch.Tensor,
    dst_block: torch.Tensor,
    dst_offset: torch.Tensor,
    cache_tensors: tuple[torch.Tensor, ...],
    *,
    num_kv_heads: int,
    head_dim: int,
    group_size: int,
    bits: int,
    valid: torch.Tensor | None = None,
) -> None:
    """Quantize ``len(src_row)`` K/V groups into packed paged-cache storage.

    ``src_strides`` gives the (row, head, position) element strides of
    ``k_src``/``v_src`` (``v_src`` must share them; the channel stride must be
    1). ``cache_tensors`` is ``(k_codes, k_scale, k_zero, v_codes, v_scale,
    v_zero)``. ``valid`` optionally skips groups whose flag is zero.
    """

    num_groups = int(src_row.numel())
    if num_groups == 0:
        return
    k_codes, k_scale, k_zero, v_codes, v_scale, v_zero = cache_tensors
    _kivi_store_groups_kernel[(num_groups, num_kv_heads)](
        k_src,
        v_src,
        src_row,
        src_pos,
        dst_block,
        dst_offset,
        valid if valid is not None else src_row,
        k_codes,
        k_scale,
        k_zero,
        v_codes,
        v_scale,
        v_zero,
        src_mod,
        src_strides[0],
        src_strides[1],
        src_strides[2],
        k_codes.stride(0),
        k_codes.stride(1),
        k_codes.stride(2),
        k_scale.stride(0),
        k_scale.stride(1),
        k_scale.stride(2),
        v_codes.stride(0),
        v_codes.stride(1),
        v_codes.stride(2),
        v_scale.stride(0),
        v_scale.stride(1),
        v_scale.stride(2),
        GROUP=group_size,
        HEAD_DIM=head_dim,
        BITS=bits,
        HAS_VALID=valid is not None,
        num_warps=4,
    )


def decode_attention(
    q: torch.Tensor,
    cache_tensors: tuple[torch.Tensor, ...],
    k_ring: torch.Tensor,
    v_ring: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    slots: torch.Tensor,
    *,
    softmax_scale: float,
    residual_length: int,
    group_size: int,
    bits: int,
    block_size: int,
    num_splits: int,
    return_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Single-token decode attention over a KIVI paged cache.

    ``q`` is ``(rows, query_heads, head_dim)``. Each row attends to positions
    ``[0, cache_seqlens[row]]``: the quantized prefix through ``block_table``
    and the residual tail through ``k_ring``/``v_ring[slots[row]]``.
    Returns ``(out, lse)`` where ``lse`` (natural log, ``(rows, query_heads)``)
    is only computed when ``return_lse`` is set.
    """

    rows, query_heads, head_dim = q.shape
    num_kv_heads = k_ring.shape[1]
    queries_per_kv = query_heads // num_kv_heads
    k_codes, k_scale, k_zero, v_codes, v_scale, v_zero = cache_tensors
    out = torch.empty_like(q)
    lse = torch.empty((rows, query_heads), device=q.device, dtype=torch.float32) if return_lse else None
    if rows == 0:
        return out, lse
    part_acc = torch.empty((rows, query_heads, num_splits, head_dim), device=q.device, dtype=torch.float32)
    part_max = torch.empty((rows, query_heads, num_splits), device=q.device, dtype=torch.float32)
    part_sum = torch.empty_like(part_max)
    _kivi_decode_partial_kernel[(rows, num_kv_heads, num_splits)](
        q,
        k_codes,
        k_scale,
        k_zero,
        v_codes,
        v_scale,
        v_zero,
        k_ring,
        v_ring,
        block_table,
        cache_seqlens,
        slots,
        part_acc,
        part_max,
        part_sum,
        softmax_scale * math.log2(math.e),
        residual_length,
        k_ring.shape[2],
        q.stride(0),
        q.stride(1),
        k_codes.stride(0),
        k_codes.stride(1),
        k_codes.stride(2),
        k_scale.stride(0),
        k_scale.stride(1),
        k_scale.stride(2),
        v_codes.stride(0),
        v_codes.stride(1),
        v_codes.stride(2),
        v_scale.stride(0),
        v_scale.stride(1),
        v_scale.stride(2),
        k_ring.stride(0),
        k_ring.stride(1),
        k_ring.stride(2),
        block_table.stride(0),
        part_acc.stride(0),
        part_acc.stride(1),
        part_acc.stride(2),
        part_max.stride(0),
        part_max.stride(1),
        QUERIES_PER_KV=queries_per_kv,
        HEADS_PAD=max(16, triton.next_power_of_2(queries_per_kv)),
        HEAD_DIM=head_dim,
        GROUP=group_size,
        BITS=bits,
        BLOCK_SIZE=block_size,
        NUM_SPLITS=num_splits,
        num_warps=4,
    )
    _kivi_combine_kernel[(rows, query_heads)](
        part_acc,
        part_max,
        part_sum,
        out,
        lse if lse is not None else part_max,
        part_acc.stride(0),
        part_acc.stride(1),
        part_acc.stride(2),
        part_max.stride(0),
        part_max.stride(1),
        out.stride(0),
        out.stride(1),
        lse.stride(0) if lse is not None else 0,
        HEAD_DIM=head_dim,
        NUM_SPLITS=num_splits,
        SPLITS_PAD=triton.next_power_of_2(num_splits),
        HAS_LSE=lse is not None,
        num_warps=4,
    )
    return out, lse
