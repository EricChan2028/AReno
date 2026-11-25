"""Time the dense kernels behind a prefill chunk that extends a cached prefix.

One layer of Qwen3-0.6B geometry (16 query heads, 8 KV heads, head dim 128):
a chunk of NEW tokens after OLD cached tokens. Compares the native kernel the
KIVI prefix path calls today (areno_causal_attention with a query offset) with
the tiled kernel that the BF16 native backend uses for the same chunk.
"""

import time

import torch

from areno.accel.attention import areno_causal_attention
from areno.accel.kernels.native_attention import tiled_forward

torch.manual_seed(0)
dev = torch.device("cuda")
H, HK, D, PAGE = 16, 8, 128, 256
scale = D**-0.5


def bench(fn, iters=5):
    fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000, out


for old, new in [(92, 8008), (4096, 4096), (6000, 2100)]:
    total = old + new
    q = torch.randn(new, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(total, HK, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(total, HK, D, device=dev, dtype=torch.bfloat16)

    def native_offset():
        rep = H // HK
        out = areno_causal_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.repeat_interleave(rep, dim=1).transpose(0, 1).unsqueeze(0),
            v.repeat_interleave(rep, dim=1).transpose(0, 1).unsqueeze(0),
            query_start=old,
            softmax_scale=scale,
            force_native=True,
        )
        return out.squeeze(0).transpose(0, 1)

    blocks = -(-total // PAGE)
    k_pages = torch.zeros(blocks * PAGE, HK, D, device=dev, dtype=torch.bfloat16)
    v_pages = torch.zeros_like(k_pages)
    k_pages[:total] = k
    v_pages[:total] = v
    k_pages = k_pages.view(blocks, PAGE, HK, D)
    v_pages = v_pages.view(blocks, PAGE, HK, D)
    table = torch.arange(blocks, device=dev, dtype=torch.int32).view(1, -1)
    bounds = torch.tensor([0, new], device=dev, dtype=torch.int32)
    prefix = torch.tensor([old], device=dev, dtype=torch.int32)

    def tiled_paged():
        return tiled_forward(q, k_pages, v_pages, bounds, -1, scale, table=table, prefix=prefix, value_dim=D)

    t_native, a = bench(native_offset)
    t_tiled, b = bench(tiled_paged)
    diff = (a.float() - b.float()).abs().max().item()
    print(f"old={old:5d} new={new:5d}: areno_causal_attention {t_native:8.2f} ms | tiled paged {t_tiled:6.2f} ms | max |diff| {diff:.2e}")
