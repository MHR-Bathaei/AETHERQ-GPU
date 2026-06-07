#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iomanip>

// Include your actual legacy weights!
#include "model_weights.h"

#define CHECK_CUDA(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Fixed dimensions from Layer 1 (GCN1) of your model
const int NUM_NODES = 8;     // Rows of X
const int IN_FEATURES = 6;   // Cols of X
const int OUT_FEATURES = 64; // Output dimension for GCN1

// ============================================================================
// TRUE AETHERQ BATCHED KERNEL (GCN LAYER 1)
// Shape: Y[8, 64] = X[8, 6] * W[64, 6]^T + b[64]
// ============================================================================
__global__ void aetherqBatchedGCN1(const float* d_X, const float* d_W, const float* d_b, float* d_Y, int numIterations) {
    int batchIdx = blockIdx.x; 

    if (batchIdx < numIterations) {
        // Map 8x64 threads directly to our output matrix shape
        int row = threadIdx.y; // 0 to 7 (Nodes)
        int col = threadIdx.x; // 0 to 63 (Output Features)

        if (row < NUM_NODES && col < OUT_FEATURES) {
            float accum = 0.0f;
            
            // Compute Dot Product
            for (int k = 0; k < IN_FEATURES; ++k) {
                // X memory layout: [8 x 6]
                float x_val = d_X[row * IN_FEATURES + k];
                
                // W memory layout: [64 x 6]. We read it transposed.
                float w_val = d_W[col * IN_FEATURES + k];
                
                accum += x_val * w_val;
            }

            // Add the layer bias for this specific output feature
            accum += d_b[col];

            // Calculate VRAM offset for this specific batch iteration's output
            int outputOffset = batchIdx * (NUM_NODES * OUT_FEATURES);
            d_Y[outputOffset + (row * OUT_FEATURES + col)] = accum;
        }
    }
}

int main() {
    std::cout << "====================================================" << std::endl;
    std::cout << "🛰️ AETHERQ-GPU: TRUE LEGACY INTEGRATION (GCN1)" << std::endl;
    std::cout << "====================================================" << std::endl;

    const int iterations = 5000;
    
    // 1. Generate the same mock input used in your benchmarker.cpp
    std::vector<float> h_X(NUM_NODES * IN_FEATURES);
    for (int i = 0; i < NUM_NODES * IN_FEATURES; ++i) {
        h_X[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    size_t x_bytes = NUM_NODES * IN_FEATURES * sizeof(float);
    size_t w_bytes = OUT_FEATURES * IN_FEATURES * sizeof(float);
    size_t b_bytes = OUT_FEATURES * sizeof(float);
    size_t y_bytes = iterations * NUM_NODES * OUT_FEATURES * sizeof(float);

    std::cout << "Target Layout: " << iterations << " concurrent GNN Layer 1 pipelines." << std::endl;
    std::cout << "Loading actual W_gcn1 [64x6] and b_gcn1 [64] from model_weights.h..." << std::endl;

    // 2. Allocate VRAM
    float *d_X = nullptr, *d_W = nullptr, *d_b = nullptr, *d_Y = nullptr;
    CHECK_CUDA(cudaMalloc(&d_X, x_bytes));
    CHECK_CUDA(cudaMalloc(&d_W, w_bytes));
    CHECK_CUDA(cudaMalloc(&d_b, b_bytes));
    CHECK_CUDA(cudaMalloc(&d_Y, y_bytes));

    // 3. Transfer Inputs, Real Weights, and Real Biases to GPU
    CHECK_CUDA(cudaMemcpy(d_X, h_X.data(), x_bytes, cudaMemcpyHostToDevice));
    
    // Notice we are passing your actual legacy arrays (W_gcn1 and b_gcn1) directly!
    CHECK_CUDA(cudaMemcpy(d_W, W_gcn1, w_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b_gcn1, b_bytes, cudaMemcpyHostToDevice));

    // 4. Launch Configuration: 5000 Blocks, each containing 8x64 threads (512 threads per block)
    dim3 threadsPerBlock(OUT_FEATURES, NUM_NODES); 
    int blocksPerGrid = iterations;

    cudaEvent_t start, stop;
    float elapsedTimeMs = 0.0f;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::cout << "\n Firing true integrated batch..." << std::endl;
    
    CHECK_CUDA(cudaEventRecord(start));
    // Execute Layer 1 projection
    aetherqBatchedGCN1<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_W, d_b, d_Y, iterations);
    CHECK_CUDA(cudaEventRecord(stop));
    
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&elapsedTimeMs, start, stop));

    // 5. Retrieve output
    std::vector<float> h_Y(iterations * NUM_NODES * OUT_FEATURES, 0.0f);
    CHECK_CUDA(cudaMemcpy(h_Y.data(), d_Y, y_bytes, cudaMemcpyDeviceToHost));

    std::cout << "Output Shape [8x64] across 5,000 batches recovered." << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "GPU GCN1 Processing Time: " << elapsedTimeMs << " ms" << std::endl;
    std::cout << "====================================================" << std::endl;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_W));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_Y));

    return 0;
}