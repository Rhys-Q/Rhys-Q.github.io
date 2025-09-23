---
layout: distill
title: triton tutorials
date: 2025-09-14 00:00:00-0000
description: triton tutorials
tags: hpc kernel triton
giscus_comments: true
categories: hpc
featured: true
related_posts: false
---

# Vector Addition

https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#sphx-glr-getting-started-tutorials-01-vector-add-py



```python
import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)
```

Functions decorated with `@triton.jit`(such as `add_kernel` in the above code) are not directly executed as Python code, but are captured by the triton compiler and dynamically generate optimized machine code for specific hardware (such as NVIDIA GPU) at runtime.

The compilation process is triggered when the function is first called, rathen than when it is defined.This allows the compiler to generate better code based on actual input parameters such as BLOCK_SIZE and other meta parameters.

```python
pid = tl.program_id(axis=0)
```

`pid` is the program id. Triton adopts `SPMD` program model.Every "program" deal sub dataset of data.

`tl.load` copy data from DRAM (global memory ) to SRAM(shared memory).

`tl.store` copy data from SRAM to DRAM.

User dont't need to care how to move data, triton can optimize the move method automatically.

The `BLOCK_SIZE: tl.constexpr` in above code is metadata, it can be specified when call.

Triton use `tl.arange`to implement SIMD-style parallel.



Leetgpu: https://leetgpu.com/challenges/vector-addition



# Fused Softmax

The shape of X is [n_row, n_col]

the number of programs is step, so every program deal n_row / step rows.

```python
row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
```

note: `other` parameter means the value when mask = False.

`num_stages` means pipeline parallelism.

The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`



**how to decide the number of program?**

1. get the resources needed by every program

```python
kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages, num_warps=num_warps, grid=(1, ))
kernel._init_handles()
n_regs = kernel.n_regs
size_smem = kernel.metadata.shared
...
occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
occupancy = min(occupancy, SIZE_SMEM // size_smem)
num_programs = NUM_SM * occupancy
num_programs = min(num_programs, n_rows)
```

there are two kinds of resource:

- shared memory (L1 Cache)
- registers

2. Divide the system resources by the resources required by the program, and take the minimum value.

|               | System resources  | Need by every program                                                                                                                                     |
| ------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Register      | NUM_SM* NUM_REGS  | n_regs * WARP_SIZE * num_warps (a program is a thread-block in coda, every thread-block has WARP_SIZE * num_warps threads, and a thread need n_regs regs) |
| shared memory | NUM_SM* SIZE_SMEM | size_smem (all threads in a thread-block share the same shared-memory)                                                                                    |

![image-20250915234227016](https://raw.githubusercontent.com/Rhys-Q/mypic/img/picgo/image-20250915234227016.png)



# Matrix Mutiplication

![image-20250917002724000](https://raw.githubusercontent.com/Rhys-Q/mypic/img/picgo/image-20250917002724000.png)

```
# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
```

the shape of `offs_am[:, None]` is [BLOCK_SIZE_M, 1], while the shape of `offs_k[None, :]`is [1, BLOCK_SIZE_N].

so after broadcasting, the shape of `a_ptrs` is [BLOCK_SIZE_M, BLOCK_SIZE_N].

 

# Low-Memory Dropout

At evaluation time we want to use the full power of the network so we set p=0. Naively this would increase the norm of the output (which can be a bad thing, e.g. it can lead to artificial decrease in the output softmax temperature). To prevent this we multiply the output by 1/(1-p) , which keeps the norm consistent regardless of the dropout probability.



```python
@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output


x = torch.randn(size=(10, ), device=DEVICE)
# Compare this to the baseline - dropout mask is never instantiated!
output = seeded_dropout(x, p=0.5, seed=123)
output2 = seeded_dropout(x, p=0.5, seed=123)
output3 = seeded_dropout(x, p=0.5, seed=512)

print(
    tabulate.tabulate([
        ["input"] + x.tolist(),
        ["output (seed = 123)"] + output.tolist(),
        ["output (seed = 123)"] + output2.tolist(),
        ["output (seed = 512)"] + output3.tolist(),
    ]))
```

Triton already support random!

# Layer Normalization


# Layer Normalization






$$
y = \frac{x-E[x]}{\sqrt{Var(x)+\epsilon}} * w +b
$$
one block deal one row!

