---
layout: distill
title: Implement An Attention with BSHD Layout Using Triton
date: 2025-10-22 00:00:00-0000
description: implement an attention with bshd layout using triton
tags: hpc triton attention
giscus_comments: true
categories: hpc
featured: true
related_posts: false
---



# Layout of Attention

Normally, there are two layouts of attention:

1. BSHD, means that the shape of Q, K, V is [batch_size, seq_len, num_head, head_dim].
2. BHSD, means that the shape of Q, K, V is [batch_size, num_head, head_dim].

[flash attention 2](https://arxiv.org/pdf/2307.08691) tile the seq_len dim, so BHSD layout is more convenient to implement  flash attenton 2. So the tutorial of triton also use the BHSD layout.

Detailed see [Fused Attention in Triton](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).

![image-20251022070324716](https://raw.githubusercontent.com/Rhys-Q/mypic/img/picgo/image-20251022070324716.png)



However, frameworks like FlashAttention, Flashinfer all use BSHD layout. So I will implement a BSHD layout attention with triton.



# BSHD Layout Attention in Triton

The main difference is the load of Q, K, V.

**Because we need to tile the seq_len dim, so we need to make Q, K, V as a three-dim Tensor.**

The new shape of Q, K, V is [y_dim, H, HEAD_DIM]

y_dim is tiled by BLOCK_M or BLOCK_N.

```python
import logging
from typing import List

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    logging.warning("Triton is not available. Install Triton for GPU kernel support.")

import tvm
from tvm.script import ir as I
from tvm.script import tir as T

# from .triton_registry import register_triton_kernel, make_tir_function_name, is_kernel_registered


logger = logging.getLogger(__name__)


# triton_attention.py
import torch
import triton
import triton.language as tl

# BLOCK size in the head-dimension for vectorized loads/stores
BLOCK_D = 64  # 调整以取得更好性能 / 受硬件 register 限制影响


# ---------------- single helper: put this before any @triton.jit kernels ----------------
@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    """
    Create or return an existing triton.tensor_descriptor.
    Must be decorated with @triton.jit so tl.make_tensor_descriptor runs in JIT.
    """
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    if strides[-1] != 1:
        # early guard; Triton expects last stride == 1
        raise ValueError("stride[-1] must be 1 for tl.make_tensor_descriptor")
    return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


# -------------------------------------------------------------------------------------


# ---------------- inner kernel: handles blocks of positions (pos)  ----------------
@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,  # [BLOCK_M, HEAD_DIM] loaded by caller
    desc_k,
    desc_v,
    offset_y,  # base row index for pos==0 for this (z,h): r_base = z*(N_CTX*H) + h
    off_h,
    dtype: tl.constexpr,
    start_m,
    qk_scale,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    H: tl.constexpr,  # number of heads (H) passed as constexpr
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    # lo/hi are positions (pos units)
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX

    # To map pos -> row index: row = z*(N_CTX*H) + pos*H + h
    # offset_y is r_base for pos==0. For pos==lo, add lo * H.
    offsetk_y = offset_y + lo
    offsetv_y = offset_y + lo

    # loop over blocks of positions
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # load k as [BLOCK_N, HEAD_DIM] and compute qk
        k = desc_k.load([offsetk_y, off_h, 0])  # returns [BLOCK_N, 1, HEAD_DIM]
        # q: [BLOCK_M, HEAD_DIM]; k.T: [HEAD_DIM, BLOCK_N]
        k1 = tl.reshape(k, [BLOCK_N, HEAD_DIM])
        q1 = tl.reshape(q, [BLOCK_M, HEAD_DIM])
        qk = tl.dot(q1, k1.T)

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(
            qk
        )  # using exp2 since qk already scaled to log2 domain / optimization
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)

        # scale accumulator by alpha
        if warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
        else:
            acc = acc * alpha[:, None]

        # load v ensuring shape [BLOCK_N, HEAD_DIM] for dot
        if dtype == tl.float8e5:
            # fp8 path (special transposed storage on some hw)
            v = desc_v.load([0, offsetv_y]).T
        else:
            v = desc_v.load([offsetv_y, off_h, 0])
            v1 = tl.reshape(v, [BLOCK_N, HEAD_DIM])

        p = p.to(dtype)
        acc = tl.dot(p, v1, acc)

        # update statistics
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        # advance offsets by BLOCK_N positions -> row += BLOCK_N * H
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


# -------------------------------------------------------------------------------------


NUM_STAGES_OPTIONS = [2, 3, 4]
configs = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BM in [64, 128]
    for BN in [32, 64, 128]
    for s in NUM_STAGES_OPTIONS
    for w in [4, 8]
]
# debug
configs = [
    triton.Config(dict(BLOCK_M=128, BLOCK_N=64), num_stages=2, num_warps=4),
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (
        torch.cuda.get_device_capability()[0] == 9
        and BLOCK_M * BLOCK_N < 128 * 128
        and conf.num_warps == 8
    )


def prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]

    # Filter out configs where BLOCK_M > N_CTX
    return [conf for conf in configs if conf.kwargs.get("BLOCK_M", 0) <= N_CTX]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


# ---------------- outer kernel: creates descriptors consistent with [Z,S,H,D] ----------------
@triton.autotune(
    configs=list(filter(keep, configs)),
    key=["N_CTX", "HEAD_DIM", "warp_specialize"],
    prune_configs_by={"early_config_prune": prune_invalid_configs},
)
@triton.jit
def _attn_fwd(
    sm_scale,
    M,
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    """
    Kernel expects input tensors in contiguous PyTorch layout [Z, S, H, D]
    (S==N_CTX). We create a 2D view [rows, HEAD_DIM] where rows enumerate (z, pos, h):
      row = z*(N_CTX*H) + pos*H + h
    and stride_row = HEAD_DIM (each row contains HEAD_DIM elements).
    When the kernel moves by BLOCK_N positions it must advance row index by BLOCK_N*H.
    """
    dtype = tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * N_CTX
    h_dim = H

    # For this mapping row stride is HEAD_DIM, last stride must be 1
    stride_y = HEAD_DIM * H
    stride_h = HEAD_DIM
    stride_col = 1

    desc_q = _maybe_make_tensor_desc(
        desc_q,
        shape=[y_dim, h_dim, HEAD_DIM],
        strides=[stride_y, stride_h, 1],
        block_shape=[BLOCK_M, 1, HEAD_DIM],
    )

    desc_v = _maybe_make_tensor_desc(
        desc_v,
        shape=[y_dim, h_dim, HEAD_DIM],
        strides=[stride_y, stride_h, 1],
        block_shape=[BLOCK_N, 1, HEAD_DIM],
    )
    desc_k = _maybe_make_tensor_desc(
        desc_k,
        shape=[y_dim, h_dim, HEAD_DIM],
        strides=[stride_y, stride_h, 1],
        block_shape=[BLOCK_N, 1, HEAD_DIM],
    )
    desc_o = _maybe_make_tensor_desc(
        desc_o,
        shape=[y_dim, h_dim, HEAD_DIM],
        strides=[stride_y, stride_h, 1],
        block_shape=[BLOCK_M, 1, HEAD_DIM],
    )

    # base row index for pos==0 at this (z, h)
    # r_base = z*(N_CTX*H) + h
    offset_y = off_z * N_CTX + start_m

    # q block starts at pos = start_m * BLOCK_M; each pos corresponds to H rows -> multiply by H
    qo_offset_y = offset_y + start_m * BLOCK_M * H

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # initial accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # qk scaling to log2 domain for exp2
    qk_scale = sm_scale * 1.44269504  # 1/log(2)

    # load q block (BLOCK_M rows starting at qo_offset_y)
    q = desc_q.load([offset_y, off_h, 0])

    # stage1/stage2 (pass H so inner can advance row by BLOCK_N*H)
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            desc_k,
            desc_v,
            offset_y,
            off_h,
            dtype,
            start_m,
            qk_scale,
            BLOCK_M,
            HEAD_DIM,
            H,
            BLOCK_N,
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX,
            warp_specialize,
        )
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            desc_k,
            desc_v,
            offset_y,
            off_h,
            dtype,
            start_m,
            qk_scale,
            BLOCK_M,
            HEAD_DIM,
            H,
            BLOCK_N,
            2,
            offs_m,
            offs_n,
            N_CTX,
            warp_specialize,
        )

    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    # m_ptrs = M + off_z * N_CTX * H + offs_m * H + off_h
    # tl.store(m_ptrs, m_i)
    acc1 = tl.reshape(acc.to(dtype), [BLOCK_M, 1, HEAD_DIM])
    desc_o.store([offset_y, off_h, 0], acc1)


# -------------------------------------------------------------------------------------


# ---------------- Host wrapper ----------------
# ---------------- Host wrapper: expects inputs in [Z, S, H, D] ----------------
def attention_forward(q, k, v, causal, sm_scale, warp_specialize=True):
    """
    Public wrapper: accepts q,k,v in layout [Z, S, H, D] (b,s,h,d).
    Kernel also expects contiguous [Z,S,H,D] and will read directly.
    Returns output o in same layout [Z, S, H, D].
    """
    # shape checks
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    Z, S, H, Dq = q.shape
    _, S2, Hk, Dk = k.shape
    _, S3, Hv, Dv = v.shape
    assert S == S2 == S3
    assert H == Hk == Hv
    assert Dq == Dk == Dv
    HEAD_DIM = Dq
    assert HEAD_DIM in {16, 32, 64, 128, 256}

    # allocate output in same layout
    o = torch.empty_like(q)
    stage = 3 if causal else 1

    # M shape matches original expectation (Z, H, S) ordering for storage of m
    M = torch.empty((Z, S, H), device=q.device, dtype=torch.float32)

    # pass tensors as-is (must be contiguous in [Z,S,H,D])
    desc_q = q
    desc_k = k
    desc_v = v
    desc_o = o

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    def grid(META):
        # dim0: blocks across sequence length (S)
        return (triton.cdiv(S, META["BLOCK_M"]), Z * H, 1)

    _attn_fwd[grid](
        sm_scale,
        M,
        Z,
        H,
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        N_CTX=S,
        HEAD_DIM=HEAD_DIM,
        STAGE=stage,
        warp_specialize=warp_specialize,
    )

    return o


def causal_attention_bs_hd(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
):
    import torch.nn.functional as F

    batch_size, seq_len_q, num_heads, head_dim = query.shape
    seq_len_k = key.shape[1]
    seq_len_v = value.shape[1]
    assert seq_len_k == seq_len_v, "key 和 value 的序列长度必须相等"

    q = query.permute(0, 2, 1, 3)  # [b, h, s_q, d]
    k = key.permute(0, 2, 1, 3)  # [b, h, s_k, d]
    v = value.permute(0, 2, 1, 3)  # [b, h, s_v, d]

    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)

    if is_causal:
        assert seq_len_q == seq_len_k, "因果掩码要求 query 和 key 的序列长度相等"
        causal_mask = torch.tril(
            torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=query.device)
        )
        attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))

    if attn_mask is not None:
        attn_scores = attn_scores + attn_mask

    attn_weights = F.softmax(attn_scores, dim=-1)

    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

    output = torch.matmul(attn_weights, v)
    output = output.permute(0, 2, 1, 3)  # 转回 [b, s, h, d]

    return output, attn_weights


# -------------------------------------------------------------------------------------
def test_attention(Z=1, H=2, N_CTX=128, HEAD_DIM=64, causal=True, dtype=torch.float16):
    torch.manual_seed(0)
    import math

    # --- 1. 构造输入 ---
    q = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device="cuda")
    k = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device="cuda")
    v = torch.randn((Z, N_CTX, H, HEAD_DIM), dtype=dtype, device="cuda")

    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    # --- 2. PyTorch reference 实现 ---
    # 注意：PyTorch 默认 attention 也是 [batch, seq, head, dim]
    q_ref = q
    k_ref = k
    v_ref = v
    ref_out = torch.nn.functional.scaled_dot_product_attention(
        q_ref.permute(0, 2, 1, 3),
        k_ref.permute(0, 2, 1, 3),
        v_ref.permute(0, 2, 1, 3),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=causal,
        # softmax_scale=sm_scale,
    ).permute(0, 2, 1, 3)

    # naive implementation
    ref_out_naive, _ = causal_attention_bs_hd(q_ref, k_ref, v_ref, is_causal=causal)
    torch.testing.assert_close(ref_out_naive, ref_out, atol=1e-2, rtol=0)

    # --- 3. Triton kernel 调用 ---
    # ✅ 不再 permute 为 [Z, H, S, D]
    #   因为我们已经修改 Triton 内核支持 [Z, S, H, D]
    tri_out = attention_forward(q, k, v, causal, sm_scale, warp_specialize=True)

    # --- 4. 校验 ---
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)

    print("✅ Triton output matches reference (within tolerance)")


# ===========================
# 简单示例（随机数据）：
if __name__ == "__main__":
    test_attention(
        Z=1,
        H=2,
        N_CTX=128,
        HEAD_DIM=128,
        causal=True,
        dtype=torch.float16,
    )

```

