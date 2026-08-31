#include <cuda_runtime.h>
#include <torch/extension.h>

/**
 * Tesla Chrono-Voxel Neural Fields (CV-NF)
 * Differentiable Volume Raymarching Forward Kernel with Early Termination
 *
 * Copyright (c) 2026 Seydi Eryilmaz (@seydivakkas)
 * All Rights Reserved.
 */

__global__ void VolumeRaymarchingForwardKernel(
    const float* __restrict__ sigmas,       // [N_rays, N_samples]
    const float* __restrict__ colors,       // [N_rays, N_samples, 3]
    const float* __restrict__ deltas,       // [N_rays, N_samples]
    const float* __restrict__ z_vals,       // [N_rays, N_samples]
    float* __restrict__ out_rgb,            // [N_rays, 3]
    float* __restrict__ out_depth,          // [N_rays]
    float* __restrict__ out_opacity,        // [N_rays]
    const int n_rays,
    const int n_samples,
    const float early_stop_thresh
) {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= n_rays) return;

    float T = 1.0f; // Cumulative transmittance
    float r = 0.0f, g = 0.0f, b = 0.0f;
    float depth = 0.0f;

    int ray_offset = ray_idx * n_samples;

    for (int s = 0; s < n_samples; ++s) {
        int idx = ray_offset + s;
        float sigma = sigmas[idx];
        float delta = deltas[idx];
        float z = z_vals[idx];

        // Alpha compositing: alpha = 1 - exp(-sigma * delta)
        float alpha = 1.0f - __expf(-sigma * delta);
        float weight = T * alpha;

        int color_offset = idx * 3;
        r += weight * colors[color_offset];
        g += weight * colors[color_offset + 1];
        b += weight * colors[color_offset + 2];
        depth += weight * z;

        T *= (1.0f - alpha);

        // Hardware efficiency early ray termination
        if (T < early_stop_thresh) {
            break;
        }
    }

    out_rgb[ray_idx * 3] = r;
    out_rgb[ray_idx * 3 + 1] = g;
    out_rgb[ray_idx * 3 + 2] = b;
    out_depth[ray_idx] = depth;
    out_opacity[ray_idx] = 1.0f - T;
}

torch::Tensor volume_raymarch_cuda_forward(
    torch::Tensor sigmas,
    torch::Tensor colors,
    torch::Tensor deltas,
    torch::Tensor z_vals,
    float early_stop_thresh = 1e-4f
) {
    const int n_rays = sigmas.size(0);
    const int n_samples = sigmas.size(1);

    auto out_rgb = torch::zeros({n_rays, 3}, sigmas.options());
    auto out_depth = torch::zeros({n_rays}, sigmas.options());
    auto out_opacity = torch::zeros({n_rays}, sigmas.options());

    const int threads = 256;
    const int blocks = (n_rays + threads - 1) / threads;

    VolumeRaymarchingForwardKernel<<<blocks, threads>>>(
        sigmas.data_ptr<float>(),
        colors.data_ptr<float>(),
        deltas.data_ptr<float>(),
        z_vals.data_ptr<float>(),
        out_rgb.data_ptr<float>(),
        out_depth.data_ptr<float>(),
        out_opacity.data_ptr<float>(),
        n_rays,
        n_samples,
        early_stop_thresh
    );

    return torch::cat({out_rgb, out_depth.unsqueeze(-1), out_opacity.unsqueeze(-1)}, -1);
}
