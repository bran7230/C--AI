# C++ Neural Network & Transformer Engine (Scratch-Built AI)

This is a low-level, high-performance C++ implementation of a neural network and transformer architecture built entirely from scratch. It focuses on learning, reasoning, and generating text without relying on machine learning libraries — enabling **full control and a deep understanding** of how models like GPT function internally.

---

## Features

### Math Engine (SIMD + OpenMP + CUDA-ready)
- ReLU activation (scalar, matrix, and SIMD-accelerated)
- Sigmoid activation (scalar & batch)
- Softmax (scalar & batch)
- Matrix multiplication (`matmul`) with SIMD & OpenMP
- Dot product (SIMD)
- Element-wise vector operations
- **OpenMP multithreading** enabled for:
  - `matmul()`
  - `relu()`
  - `sigmoidBatch()`
  - `softmaxBatch()`
  - `sigmoidDerivative()`
  - `cross_entropy(batch)`
  - `binary_cross_entropy(batch)`
- **SIMD optimizations (SSE)**
  - Accelerated ReLU, dot product, softmax
- **CUDA integration in progress**
  - Successfully compiled and ran first GPU kernel
  - Starting conversion of ReLU and `matmul` to CUDA kernels

---

### Neural Network Components
- One-hot token encoding
- Manual forward pass using `matmul + activation`
- Loss functions:
  - Categorical cross-entropy
  - Binary cross-entropy
- Derivative functions (sigmoid, softmax gradients)
- Delta weight calculation (`computeDW`)
- Manual memory structure and math-focused logic
- Backprop-ready computation flows

---

## Performance Benchmarks

Tested on AMD Ryzen 9 5900X + RTX 4070 Super (SIMD + OpenMP only — CUDA benchmarks coming soon)

| Operation                | Size                | Time (C++ SIMD + OpenMP) |
|--------------------------|---------------------|---------------------------|
| ReLU (SIMD)              | 1,000,000 x 512     | ~0.32 seconds             |
| Softmax Batch            | 1,000,000 x 512     | ~0.36 seconds             |
| Matrix Multiply (SIMD)   | 2048 x 2048         | ~0.096 seconds            |
| Matrix Multiply (OpenMP) | 2048 x 2048         | ~1.17 seconds (pre-SIMD)  |

---

## Transformer Components (In Progress)
- Token embedding + vocabulary mapping
- Positional encoding (sin/cos style)
- Dot-product self-attention
- Multi-head attention
- Feed-forward layers
- Layer normalization
- Multi-token generation logic
- Full training loop (manual gradients, backprop)

---

## Goals

- Build a GPT-style model using **pure C++**
- Add GPU acceleration with **CUDA kernels**
- Achieve transparency in how every component is computed
- Provide SIMD, OpenMP, and CUDA benchmarking
- Serve as the foundation for a reasoning, learning AI assistant named **Ada**
- Eventually reimplement everything in raw **C** for max performance & control

---

## Tech Stack

- **C++17** (manual math, no ML libs)
- **OpenMP** for multithreaded matrix & vector ops
- **SIMD (SSE)** for optimized low-level operations
- **CUDA 12.8** (nvcc support, device-side kernels in development)
- Optional: pybind11 (Python bridge for future use)

---

## How to Build & Run (CPU Optimized)

```bash
g++ -fopenmp -O3 Code/Ada.cpp -o Ada.exe
./Ada.exe

