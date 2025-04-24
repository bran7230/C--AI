# C++ Neural Network & Transformer Engine (Scratch-Built AI)

This is a low-level, high-performance C++ implementation of a neural network and transformer architecture built entirely from scratch. It focuses on learning, reasoning, and generating text without relying on ML libraries — enabling **full control and true understanding** of how models like GPT function.

---

## Features

### Math Engine (Fully Manual, Optimized)
- ReLU activation (scalar & matrix)
- Sigmoid activation (scalar & matrix)
- Softmax (scalar & batch)
- Matrix multiplication (`matmul`)
- Dot product (coming soon)
- Element-wise ops
- **OpenMP multithreading support for:**
  - `matmul()`
  - `relu()`
  - `sigmoid(matrix)`
  - `softmaxBatch()`
  - `sigmoidDerivative()`
  - `cross_entropy(batch)`
  - `binary_cross_entropy(batch)`
- **CUDA acceleration support** (Tiled Shared Memory + `float4` + loop unrolling)

### Neural Network Components
- One-hot encoding
- Gradient computation
- Delta weight calculation (`computeDW`)
- Manual memory layout for performance
- Backprop-ready activation flows
- Binary and categorical loss functions

### Performance Benchmarks (Updated)
- ReLU on 1M x 512 matrix (SIMD + OpenMP): **~0.32s**
- Softmax on 1M x 512 matrix (SIMD + OpenMP): **~0.36s**
- CPU Matrix multiply (2048 x 2048): **~1.17s**
- CUDA Matrix multiply (2048 x 2048): **~0.15s**
- CUDA Matrix multiply (4096 x 4096): **~0.042s**
- CUDA Matrix multiply (16384 x 16384): **~0.023s** (float4 tiled + unrolled kernel)

---

## Transformer Components (Planned)
- Token embedding & vocab mapping
- Positional encoding (sin/cos)
- Attention mechanism (dot-product self-attention)
- Multi-head attention
- Layer normalization
- Multi-token generation
- Full forward/backward training loop

---

## Goals

- Build a GPT-style AI model in **pure C++**
- Achieve full transparency & control over all model components
- Benchmark against existing models like GPT-2
- Add **CUDA** acceleration for GPU compute
- Eventually rewrite in raw **C** as the final optimization challenge
- Use this as the foundation for "Ada" — a learning, reasoning AI assistant

---

## Tech Stack

- **C++17**
- Standard Library only (no ML libs)
- **OpenMP** (for CPU-level parallelism)
- **CUDA** (for GPU acceleration)
- Optional: pybind11 (for Python interop)

---

## How to Build & Run

```bash
g++ -fopenmp -O3 Code/Ada.cpp -o Ada.exe
./Ada.exe
```

For CUDA:
```bash
nvcc -O3 Code/main.cu -o Ada.exe
./Ada.exe