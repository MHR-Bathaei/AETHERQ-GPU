# AETHERQ-GPU: High-Performance Batched GNN Inference Engine

A CUDA-accelerated implementation of the AETHERQ Graph Convolutional Network (GCN) projection layer, designed to explore high-throughput graph inference on NVIDIA GPUs.

This project transitions a CPU-based AVX2 implementation to a massively parallel CUDA execution model, enabling thousands of independent GNN layer evaluations to be processed concurrently.

---

## Overview

The original AETHERQ implementation used manually vectorized AVX2 instructions on the CPU to accelerate matrix operations. While efficient for individual executions, the workload remained fundamentally sequential when processing large numbers of iterations.

AETHERQ-GPU investigates an alternative execution strategy: mapping independent GNN projection workloads across the GPU execution grid and executing them simultaneously.

### CPU Version

* Sequential execution loop
* AVX2/FMA vectorization (`_mm256_fmadd_ps`)
* Optimized for low-latency single-instance execution

### GPU Version

* CUDA-based parallel execution
* One CUDA block assigned to each independent workload instance
* Optimized for throughput across large batches

---

## Benchmark Configuration

### Hardware

* GPU: NVIDIA GeForce RTX 4060 Laptop GPU
* Architecture: Ada Lovelace
* Compute Capability: 8.9 (SM 89)
* L2 Cache: 32 MB

### Workload

* Total Iterations: 5,000
* Input Shape: 8 × 6
* Weight Shape: 64 × 6
* Output Shape: 8 × 64

Each iteration performs an independent GCN projection using identical layer dimensions.

---

## Performance Results

| Implementation        | Total Runtime | Effective Time per Iteration | Notes                                  |
| --------------------- | ------------- | ---------------------------- | -------------------------------------- |
| AVX2 CPU (Sequential) | 20.500 ms     | 4.10 µs                      | Strong single-instance latency         |
| CUDA GPU (Batched)    | 0.211 ms      | 0.042 µs (42 ns)             | High throughput via parallel execution |

### Observed Speedup

For this benchmark configuration:

```text
Speedup = 20.500 ms / 0.211 ms ≈ 97.1×
```

The result illustrates the benefit of executing thousands of independent projection workloads concurrently on the GPU rather than processing them sequentially on the CPU.

---

## Computation

The kernel performs a fused linear projection and bias addition:

```text
Y = X · Wᵀ + b
```

Where:

* X = input node-feature matrix
* W = learnable projection weights
* b = bias vector
* Y = projected output matrix

---

## CUDA Execution Layout

### Grid Configuration

```text
Grid:
    N blocks
    N = number of benchmark iterations
```

Each CUDA block processes one independent projection workload.

### Block Configuration

```text
Threads per block:
    [OUT_FEATURES, NUM_NODES]
    [64, 8]
```

This maps the thread topology directly to the output matrix dimensions.

### Memory Strategy

* Coalesced global memory access patterns
* Shared static weight and bias tensors across workloads
* Register-based accumulation for projection computation
* GPU-wide parallel execution of independent inference instances

---

## Technical Skills Demonstrated

* CUDA kernel development
* Thread-block and grid design
* GPU memory hierarchy awareness
* Performance benchmarking
* Throughput vs latency analysis
* AVX2 SIMD optimization
* CPU/GPU architectural comparison

---

## Build Instructions

Compile using NVIDIA CUDA Compiler (`nvcc`):

```powershell
nvcc -O3 -arch=sm_89 -I./eigen aetherq_gpu_benchmark.cu -o aetherq_gpu_benchmark.exe
```

Run the benchmark:

```powershell
.\aetherq_gpu_benchmark.exe
```

---

## Project Purpose

This project explores practical GPU acceleration techniques for graph neural network workloads and serves as a study of:

* CUDA programming
* GPU execution topology
* Throughput-oriented inference systems
* CPU vs GPU performance characteristics
* Parallel execution strategies for graph workloads

The implementation focuses on understanding how workload mapping, thread organization, and memory access patterns influence inference performance on modern NVIDIA GPUs.
