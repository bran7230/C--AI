# C++(AND CUDA) Neural Network & Transformer Engine(OPEN SOURCE)

This is a low-level, high-performance C++ implementation of a neural network and transformer architecture built entirely from scratch.(ITS ALSO OPEN SOURCE SO FEEL FREE TO TAKE CODE ETC) It focuses on learning, reasoning, and generating text without relying on ML libraries — enabling **full control and true understanding** of how models like GPT function.
(CPU SUPPORT IS DEPRECIATED AND IS NO LONGER SUPPORTED, YOU MUST HAVE A NVIDIA GPU TO USE THE CODE)

---

## Features

### Math Engine (Fully Manual + Tensor Core Accelerated)
- ReLU activation (scalar & matrix)
- Sigmoid activation (scalar & matrix)
- Softmax (scalar & batch)
- Element-wise operations
- One-hot encoding
- Gradient computation
- Delta weight calculation (`computeDW`)
- Loss functions:
  - Cross-entropy (batch)
  - Binary cross-entropy (batch)
- **OpenMP multithreading support** for:
  - `relu()`
  - `sigmoid()`
  - `softmaxBatch()`
  - `sigmoidDerivative()`
  - `cross_entropy(batch)`
  - `binary_cross_entropy(batch)`
- **CUDA acceleration support:**
  - ReLU CUDA kernels (1D and 2D)
  - Softmax CUDA kernels (shared memory)
  - Matmul using Tensor Cores (`cublasGemmEx`)
  - Fused Matmul + Bias addition using Tensor Cores
  - Fully batched Matmul + Bias GPU-only execution (no Host-Device overhead)

---

## Neural Network Components
- Manual memory layout for performance
- Fully batched matrix multiplication support
- GPU-persistent memory optimized for training/inference
- Tensor Core accelerated Dense layers
- Future-ready for Transformer scaling
- Backpropagation-ready design

---

## Performance Benchmarks (Updated)

| Test                          | Result |
|--------------------------------|--------|
| ReLU on 1M × 512 matrix (SIMD + OpenMP) | **~0.32s** |
| Softmax on 1M × 512 matrix (SIMD + OpenMP) | **~0.36s** |
| CPU Matrix multiply (2048 × 2048, OpenMP) | **~1.17s** |
| **Tensor Core Matmul (4096 × 4096, cuBLAS)** | **~10–15ms** (~20–30 TFLOPS) |
| **Tensor Core Matmul (16384 × 16384, cuBLAS)** | **~122ms** (~72 TFLOPS ) |
| Batched Tensor Core Matmul + Bias (16384 × 16384) | **~122ms** (~72 TFLOPS) |

Tensor Core matmul achieved ~**70+ TFLOPS** performance!  
Batched fully on GPU without CPU bottlenecks.

---

## Transformer Components (In Progress)

- Token embedding & vocab mapping
- Positional encoding (sin/cos)
- Attention mechanism (dot-product self-attention)
- Multi-head attention
- Layer normalization
- Multi-token generation
- Full forward/backward training loop (planned)

---

## Goals

- Build a GPT-style AI model in **pure C++**
- Achieve full transparency & control over all model components
- Benchmark against existing models like GPT-2
- Fully leverage **Tensor Core acceleration** for matmul
- Optimize Transformer scaling for large batches
- Eventually rewrite in raw **C** as the final optimization challenge
- Use this foundation for "Syntari" — a self-learning, reasoning AI assistant

---

## Tech Stack

- **C++17**
- Standard Library only (no ML libraries)
- **OpenMP** (for CPU-level parallelism)
- **CUDA** (for GPU acceleration, Tensor Cores)
- **cuBLAS** (for fast batched matmul)
- Optional: pybind11 (for Python interop)

---
## CUDA Build
```bash
nvcc -O3 -arch=sm_70 -lcublas -Xcompiler "/openmp" -o syntari Code/main.cu Code/math/Syntari.cu
./syntari
```
---
## How to Build & Run(DEPRECIATED)

For CPU build:
```bash
g++ -fopenmp -O3 Code/Ada.cpp -o Ada.exe
./Ada.exe
