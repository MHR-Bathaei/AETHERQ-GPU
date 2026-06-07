# AETHERQ-GPU: High-Performance Batched GNN Inference Engine

A hardware-accelerated, massively parallel optimization of the **AETHERQ** Graph Convolutional Network (GCN) layer. This engine transitions the legacy framework from sequential CPU AVX2 SIMD loops to a custom-architected CUDA grid topology, unlocking extreme throughput for micro-batch telemetry processing.

---

## 🚀 The Architectural Leap

The legacy AETHERQ implementation achieved optimization via manual AVX2 vector intrinsics on the CPU. However, processing iterations sequentially introduced scaling bottlenecks when handling high-frequency sensor streams.

**AETHERQ-GPU** resolves the memory and execution walls by implementing a **Parallel Batching Layout**. Instead of dispatching thousands of sequential kernels or loops, the engine packs the entire benchmark workload into a single concurrent execution wave.

* **Legacy Engine:** Sequential CPU loops + manual vectorization (`_mm256_fmadd_ps`).
* **Upgraded Engine:** Massively parallel CUDA execution grid where each hardware block completely absorbs and executes an entire distinct GNN layer projection instance simultaneously.

---

## 📊 Performance Profile

### Execution Environment

* **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (Ada Lovelace Architecture, SM 8.9)
* **Compute Capabilities:** Dedicated L1/Shared Memory SRAM per Streaming Multiprocessor, 32 MB L2 Cache
* **Workload Configuration:** 5,000 concurrent iterations, Input Shape $[8 \times 6]$ (8 Nodes, 6 Features), Weight Projection Shape $[64 \times 6]$, Output Shape $[8 \times 64]$

### Microbenchmark Metrics (5,000 Total Iterations)

| Architecture Layer | Total Workload Runtime | Effective Per-Layer Throughput | Engineering Trade-off |
| :--- | :--- | :--- | :--- |
| **Legacy AVX2 CPU** (Sequential Loop) | 20.500 ms | 4.10 $\mu$s | **Latency-Optimized:** Ultra-fast single execution, scales poorly over large batches. |
| **AETHERQ-GPU** (Parallel Batched) | **0.211 ms** | **0.042 $\mu$s** (42 ns) | **Throughput-Optimized:** Minor kernel launch overhead, scales exceptionally across massive batches. |

**Key Insight:** While the host CPU maintains an incredible single-instance latency baseline of 4.10 $\mu$s due to immediate L1 cache proximity, it falls behind under heavy iteration scaling. By parallelizing all 5,000 iterations into a concurrent CUDA grid, AETHERQ-GPU achieves a **97.1x throughput speedup** over the sequential CPU loop.

---

## 🛠️ Core Infrastructure Architecture

### Mathematical Operations

The kernel computes a fused forward projection and bias addition for an arbitrary number of batch components concurrently:

$$Y_{\text{batch}} = X \cdot W_{\text{gcn1}}^T + b_{\text{gcn1}}$$

### Memory & Thread Topology

* **Grid Layout:** 1D Grid of $N$ Blocks (where $N = \text{Total Iterations}$).
* **Block Layout:** 2D Thread Allocation matching the precise matrix dimensions of the layer's output spatial footprint ($\text{Threads} = [\text{OUT\_FEATURES}, \text{NUM\_NODES}] = [64, 8]$).
* **Memory Alignment:** Coalesced global memory indexing maps raw static weights (`W_gcn1`) and biases (`b_gcn1`) cleanly into register spaces, optimizing memory bus utilization across the PCIe boundary.


---

## ⚡ Compilation & Deployment

The engine compiles under the NVIDIA CUDA Compiler (`nvcc`) utilizing standard host optimization flags and strict target architecture matching for Ada Lovelace runtime optimization (`sm_89`).

```powershell
# Compile the production engine pipeline
nvcc -O3 -arch=sm_89 -I./eigen aetherq_gpu_benchmark.cu -o aetherq_gpu_benchmark.exe

# Execute the high-precision profiling benchmark
.\aetherq_gpu_benchmark.exe

```
