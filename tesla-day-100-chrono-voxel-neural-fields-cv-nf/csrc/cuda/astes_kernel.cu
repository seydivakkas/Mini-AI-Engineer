#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <torch/extension.h>

namespace cg = cooperative_groups;

/**
 * Tesla Chrono-Voxel Neural Fields (CV-NF)
 * Warp-Level Asynchronous Spatio-Temporal Event Surface (ASTES) Integration Kernel
 * 
 * Computes continuous temporal exponential decay accumulation on asynchronous event streams:
 * S(x, y, t) = \sum_{k: (x_k, y_k) = (x, y)} p_k * exp(-(t - t_k) / \tau) * 1_{t >= t_k}
 *
 * Copyright (c) 2026 Seydi Eryilmaz (@seydivakkas)
 * All Rights Reserved.
 */

__global__ void ComputeASTESContinuousKernel(
    const int* __restrict__ event_x,
    const int* __restrict__ event_y,
    const float* __restrict__ event_t,
    const float* __restrict__ event_polarity,
    float* __restrict__ output_surface,
    const int num_events,
    const float current_query_time,
    const float decay_tau,
    const int width,
    const int height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_events) return;

    float t_k = event_t[idx];
    if (t_k > current_query_time) return; // Future event guard

    int x = event_x[idx];
    int y = event_y[idx];

    if (x < 0 || x >= width || y < 0 || y >= height) return;

    float p = event_polarity[idx];
    float delta_t = current_query_time - t_k;
    float decayed_weight = p * __expf(-delta_t / decay_tau);

    // Hardware-accelerated atomic accumulation into 2D continuous surface
    int surface_idx = y * width + x;
    atomicAdd(&output_surface[surface_idx], decayed_weight);
}

torch::Tensor astes_continuous_surface_cuda(
    torch::Tensor event_x,
    torch::Tensor event_y,
    torch::Tensor event_t,
    torch::Tensor event_polarity,
    float current_query_time,
    float decay_tau,
    int width,
    int height
) {
    const int num_events = event_x.size(0);
    auto output_surface = torch::zeros({height, width}, event_t.options());

    if (num_events == 0) {
        return output_surface;
    }

    const int threads = 256;
    const int blocks = (num_events + threads - 1) / threads;

    ComputeASTESContinuousKernel<<<blocks, threads>>>(
        event_x.data_ptr<int>(),
        event_y.data_ptr<int>(),
        event_t.data_ptr<float>(),
        event_polarity.data_ptr<float>(),
        output_surface.data_ptr<float>(),
        num_events,
        current_query_time,
        decay_tau,
        width,
        height
    );

    return output_surface;
}
