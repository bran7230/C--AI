# C++ Neural Network & Transformer Engine (Scratch-Built AI)

This is a low-level C++ implementation of a neural network and transformer architecture built entirely from scratch. It focuses on learning, reasoning, and generating text without relying on machine learning libraries — enabling full control and deep understanding of how models like GPT function.

---
# To build use g++ -O3 Code/Ada.cpp -o Ada.exe
## 🧠 Features (In Progress)

### ✅ Math Engine
- Manual implementation of:
  - ReLU activation (scalar & vectorized)
  - Sigmoid activation (scalar)
  - Softmax (vector)
  - Matrix multiplication (`matmul`)
  - Dot product (coming soon)
  - Element-wise vector ops

### ✅ Neural Network Foundation
- Vector-based math operations
- Full support for custom activation layers
- Manual memory management & performance considerations
- Focus on clean, modular, readable code

### ⚙️ Transformer Components (Planned)
- Token embedding & vocabulary mapping
- Positional encoding
- Attention mechanism (self-attention)
- Multi-head attention (optional)
- Layer normalization
- Multi-token autoregressive generation
- Training loop with backpropagation (manual)

---

## 💡 Goals

- Build a GPT-style AI model in **pure C++**
- Learn how every part of a neural network works, from math to memory
- Avoid black-box libraries — **understand everything**
- Eventually port to raw C as a final performance challenge
- Use this project as the foundation for an intelligent assistant with reasoning and learning capabilities

---

## 🧱 Tech Stack

- **C++17**
- Standard Library only (no external ML libraries)
- Optional: pybind11 (for Python interop, if needed later)
- Optional: CUDA (for future GPU acceleration)

---

## 🛠️ How to Build & Run

```bash
g++ -std=c++17 -O3 -Wall -o main main.cpp
./main
