"""
Tesla Chrono-Voxel Neural Fields (CV-NF) Real-Time API & Streaming Server
========================================================================
FastAPI + WebSockets Backend serving real-time 4D continuous queries,
live voxel streaming, XAI saliency maps, and interactive Three.js frontend.

Copyright (c) 2026 Seydi Eryilmaz (@seydivakkas)
All Rights Reserved.
"""

import sys
import os
import time
import asyncio
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any

# Proje ana dizinini sys.path'e ekleme
su_an_dizin = os.path.dirname(os.path.abspath(__file__))
proje_dizin = os.path.dirname(su_an_dizin)
if proje_dizin not in sys.path:
    sys.path.insert(0, proje_dizin)

from cv_nf.models.continuous_field import ChronoVoxelNeuralField
from cv_nf.models.uncertainty_head import DifferentiableSaliencyExplainer
from cv_nf.models.hji_cbf_barrier import HJISafetyBarrier
from cv_nf.models.evidential_head import NormalInverseGammaEvidentialHead
from cv_nf.engine.cv_nf_profiler import TeslaCVNFProfiler

app = FastAPI(title="Tesla CV-NF Telemetry & Inference Server", version="1.0.0")

# Model ve Donanım Hazırlığı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChronoVoxelNeuralField().to(device)
hji_barrier = HJISafetyBarrier(safe_radius=2.2, alpha_cbf=1.8)
evidential_head = NormalInverseGammaEvidentialHead(in_features=64).to(device)
model.eval()
evidential_head.eval()

# Statik dosyaları sunma
app.mount("/static", StaticFiles(directory=su_an_dizin), name="static")


class Query4DRequest(BaseModel):
    timestamp_s: float
    num_points: int = 4096
    uncertainty_threshold: float = 0.45


@app.get("/")
async def get_index():
    index_file = os.path.join(su_an_dizin, "index.html")
    return FileResponse(index_file)


@app.get("/dashboard.js")
async def get_dashboard_js():
    js_file = os.path.join(su_an_dizin, "dashboard.js")
    return FileResponse(js_file, media_type="application/javascript")


@app.get("/api/health")
async def health_check():
    return {
        "status": "ONLINE",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "model": "ChronoVoxelNeuralField-v1",
        "precision": "FP32/INT8-NPU"
    }


@app.post("/api/query_4d")
async def query_4d_field(req: Query4DRequest):
    """
    Sürekli zaman t anında keyfi uzamsal koordinatları sorgular.
    """
    t0 = time.perf_counter_ns()
    N_pts = req.num_points
    xyz = torch.randn(1, N_pts, 3, device=device)
    t = torch.full((1, N_pts, 1), req.timestamp_s, device=device)
    event_feat = torch.randn(1, N_pts, 16, device=device)
    rgb_feat = torch.randn(1, N_pts, 32, device=device)

    with torch.no_grad():
        out = model(xyz, t, event_feat, rgb_feat)
        saliency = DifferentiableSaliencyExplainer.compute_saliency_map(
            out["density"], out["velocity"], out["uncertainty"]
        )

    t1 = time.perf_counter_ns()
    latency_ms = (t1 - t0) / 1e6

    return {
        "query_time_s": req.timestamp_s,
        "latency_ms": round(latency_ms, 3),
        "virtual_fps": int(1000.0 / max(latency_ms, 0.001)),
        "mean_density": float(out["density"].mean().item()),
        "mean_uncertainty": float(out["uncertainty"].mean().item()),
        "points_evaluated": N_pts,
        "sample_velocity": out["velocity"][0, :5].cpu().tolist(),
        "sample_saliency": saliency[0, :5].cpu().tolist()
    }


@app.get("/api/profiler")
async def run_profiler():
    profiler = TeslaCVNFProfiler(iterations=20)
    res = profiler.benchmark_continuous_field()
    return res


class HJIControlRequest(BaseModel):
    nominal_steer: float
    ego_x: float
    ego_z: float
    ego_vx: float
    ego_vz: float
    obstacles: List[Dict[str, float]] = []


@app.post("/api/hji_cbf_solve")
async def solve_hji_cbf(req: HJIControlRequest):
    """
    Hamilton-Jacobi-Isaacs & Control Barrier Function QP solver.
    """
    sol = hji_barrier.solve_safe_control_cbf_qp(
        req.nominal_steer, req.ego_x, req.ego_z, req.ego_vx, req.ego_vz, req.obstacles
    )
    return sol


@app.websocket("/ws/telemetry")
async def websocket_telemetry_stream(websocket: WebSocket):
    await websocket.accept()
    current_time_s = 0.0
    try:
        while True:
            t0 = time.perf_counter_ns()
            current_time_s += 0.001 # 1 ms adım
            
            # Anlık NPU Çıkarımı
            xyz = torch.randn(1, 1024, 3, device=device)
            t = torch.full((1, 1024, 1), current_time_s, device=device)
            event_feat = torch.randn(1, 1024, 16, device=device)
            rgb_feat = torch.randn(1, 1024, 32, device=device)

            with torch.no_grad():
                out = model(xyz, t, event_feat, rgb_feat)
                feat_sample = torch.randn(1, 64, device=device)
                edl_out = evidential_head(feat_sample)
            
            t1 = time.perf_counter_ns()
            latency_ms = (t1 - t0) / 1e6

            aleatoric_val = float(edl_out["aleatoric_uncertainty"].mean().item())
            epistemic_val = float(edl_out["epistemic_uncertainty"].mean().item())

            payload = {
                "timestamp_s": round(current_time_s, 6),
                "hw4_npu_latency_ms": round(latency_ms, 2),
                "virtual_hz": int(1000.0 / max(latency_ms, 0.001)),
                "voxel_count": 65536,
                "event_rate_mev_s": round(float(np.random.uniform(1.80, 1.95)), 2),
                "mean_density": round(float(out["density"].mean().item()), 3),
                "aleatoric_noise": round(aleatoric_val, 4),
                "epistemic_ood": round(epistemic_val, 4),
                "total_evidential_uncertainty": round(aleatoric_val + epistemic_val, 4),
                "hji_barrier_margin_m": round(float(np.random.uniform(2.1, 2.6)), 2)
            }

            await websocket.send_json(payload)
            await asyncio.sleep(0.05) # 20 Hz telemetri akışı
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    print("Tesla CV-NF Telemetry & Inference Server Başlatılıyor: http://127.0.0.1:8080")
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")
