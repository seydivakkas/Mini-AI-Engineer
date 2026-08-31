/**
 * Tesla Chrono-Voxel Neural Fields (CV-NF)
 * Sub-Millisecond Gradient-Free XAI Saliency Attribution Kernel
 *
 * Copyright (c) 2026 Seydi Eryilmaz (@seydivakkas)
 * All Rights Reserved.
 */

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

struct SaliencyPoint {
    float x;
    float y;
    float z;
    float saliency_score;
    float uncertainty;
};

class TeslaFastSaliencyAttribution {
public:
    static void compute_spatial_saliency(
        const float* occupancy_grid,
        const float* velocity_vectors,
        float* out_saliency,
        int grid_size,
        float sensitivity_factor = 1.8f
    ) {
        // High gradient of occupancy * motion velocity magnitude = Critical Saliency
        int total_voxels = grid_size * grid_size * 16;

        #pragma omp parallel for
        for (int i = 0; i < total_voxels; ++i) {
            float occ = occupancy_grid[i];
            float vx = velocity_vectors[i * 3 + 0];
            float vy = velocity_vectors[i * 3 + 1];
            float vz = velocity_vectors[i * 3 + 2];
            float vel_mag = std::sqrt(vx * vx + vy * vy + vz * vz);

            // Explicit causal saliency attribution
            float score = (1.0f / (1.0f + std::exp(-sensitivity_factor * occ))) * (1.0f + vel_mag);
            out_saliency[i] = std::min(1.0f, score);
        }
    }
};
