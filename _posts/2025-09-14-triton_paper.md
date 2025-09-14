---
layout: post
title: triton paper
date: 2025-09-14 00:00:00-0000
description: read triton paper
tags: hpc kernel triton
categories: hpc
related_posts: false
---



# Paper Link

https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf



# Three Distinct Approaches for DSLs

- Tensor-level IRs:lower graph into cuda kernel using pattern matching.

- The Polyhedral Model: abstract loop into polyhedral model, and schedule loops using split, reorder, fuse, etc automaticlly, just like TVM Ansor.

- Loop Synthesizers: compute and schedule are separate.

TVM has all above features.

# The Overview of  Triton

![image-20250914191358770](https://raw.githubusercontent.com/Rhys-Q/mypic/img/picgo/image-20250914191358770.png)

- Triton-C: A C-like frontend, however, we use python frontend now.
- Triton-IR: An LLVM-based IR.
- Triton-JIT: A JIT compiler, compile Triton-IR program into llvm bitcode. Including machine-independ passes , machine-depend passes and an auto-tuner. 

# Programming Model

## Cuda Programming Model

https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/

The execution of CUDA code on GPUs is supported by an SPMD programming model.

![image-20250914191422112](https://raw.githubusercontent.com/Rhys-Q/mypic/img/picgo/image-20250914191422112.png)

## Triton Programming Model

https://openai.com/index/triton/

|                          | cuda   | Triton    |
| ------------------------ | ------ | --------- |
| Memory Coalescing        | Manual | Automatic |
| Shared Memory Management | Manual | Automatic |
| Scheduling (Within SMs)  | Manual | Automatic |
| Scheduling (Across SMs)  | Manual | Manual    |

In cuda，there are grid, block, thread. In triton, there are only program instance.

The program model of triton is SPMD + SIMD.

In an program instance, we can use SIMD-style api to load, store and calculate, triton helps us to convert to cuda warp, thread operation.

![image-20250914191437456](https://raw.githubusercontent.com/Rhys-Q/mypic/img/picgo/image-20250914191437456.png)

# Triton IR

![image-20250914191453236](https://raw.githubusercontent.com/Rhys-Q/mypic/img/picgo/image-20250914191453236.png)

Triton IR is composed of one or multiple modules.

Each modules is composed of functions, global variables, constants and other miscellaneous symbols.

Additional visibility, alignment and linkage specifiers can be added if desired. Function attributes (such as inlining hints) and parameter attributes (such as read-only, aliasing hints) can also be specified, allowing compiler backends to perform more aggressive optimizations by, for instance, making better use of read-only memory caches.

This header is followed by a body composed of a list of basic blocks whose interdependencies form the **Control Flow Graph** (CFG) of the function.

triton use PSSA(Predicated SSA) form to support tiled-level control flow. 

![image-20250914191503146](https://raw.githubusercontent.com/Rhys-Q/mypic/img/picgo/image-20250914191503146.png)



# JIT Compiler



# Numberical Experiments

