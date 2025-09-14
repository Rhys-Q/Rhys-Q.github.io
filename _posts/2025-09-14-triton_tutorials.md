---
layout: post
title: triton tutorials
date: 2025-09-14 00:00:00-0000
description: triton tutorials
tags: hpc kernel triton
categories: hpc
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

