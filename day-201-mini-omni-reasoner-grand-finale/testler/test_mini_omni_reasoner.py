"""
PyTest Birim Testleri - Day 201: Mini-Omni Reasoner v1.0 (201 GÜNLÜK BÜYÜK FİNAL).
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mini_omni_model import (
    MultimodalPatchProjector,
    TritonFusedRMSNormLayer,
    TritonFlashAttention2Block,
    SparseMoERoutingLayer,
    MiniOmniReasonerModel,
)
from src.omni_reasoning_motoru import ChainOfThoughtReasoner
from src.omni_benchmark_profilleyici import OmniBenchmarkProfilleyici
from src.gorsellestirici import OmniGrandFinaleGorsellestirici


def test_multimodal_patch_projector():
    """1. MultimodalPatchProjector görsel ve işitsel tensörleri embed_dim uzayına izdüşürmelidir."""
    proj = MultimodalPatchProjector(vision_dim=64, audio_dim=32, embed_dim=128)
    v_in = torch.randn(2, 8, 64)
    a_in = torch.randn(2, 4, 32)

    v_out = proj.project_vision(v_in)
    a_out = proj.project_audio(a_in)

    assert v_out.shape == (2, 8, 128)
    assert a_out.shape == (2, 4, 128)


def test_triton_fused_rmsnorm():
    """2. TritonFusedRMSNormLayer normalizasyon ve artık (residual) toplamını doğru hesaplamalıdır."""
    norm = TritonFusedRMSNormLayer(dim=128)
    x = torch.randn(2, 10, 128)
    res = torch.randn(2, 10, 128)

    out, new_res = norm(x, residual=res)
    assert out.shape == (2, 10, 128)
    assert new_res.shape == (2, 10, 128)


def test_triton_flash_attention_2():
    """3. TritonFlashAttention2Block nedensel maskeli dikkat çıktısını doğru boyutta üretmelidir."""
    attn = TritonFlashAttention2Block(embed_dim=128, num_heads=4)
    x = torch.randn(2, 12, 128)

    out = attn(x)
    assert out.shape == (2, 12, 128)


def test_sparse_moe_routing():
    """4. SparseMoERoutingLayer Top-2 uzman çıktısını ve kapı olasılıklarını üretmelidir."""
    moe = SparseMoERoutingLayer(embed_dim=128, num_experts=4, top_k=2)
    x = torch.randn(2, 10, 128)

    out, gate_probs = moe(x)
    assert out.shape == (2, 10, 128)
    assert gate_probs.shape == (20, 4)


def test_mini_omni_model_forward():
    """5. MiniOmniReasonerModel çok modlu girişlerle uçtan uca logits üretmelidir."""
    model = MiniOmniReasonerModel(vocab_size=500, embed_dim=128, num_layers=1)
    text = torch.randint(0, 500, (1, 8))
    vision = torch.randn(1, 4, 64)
    audio = torch.randn(1, 2, 32)

    res = model(text, vision_patches=vision, audio_patches=audio)
    assert "logits" in res
    assert res["seq_len"] == 14  # 4 vision + 2 audio + 8 text = 14
    assert res["logits"].shape == (1, 14, 500)


def test_chain_of_thought_reasoner():
    """6. ChainOfThoughtReasoner <think> etiketli CoT akıl yürütmesi ve metrik üretmelidir."""
    reasoner = ChainOfThoughtReasoner()
    res = reasoner.solve_multimodal_problem("Test problem", has_vision=True, has_audio=False)

    assert "<think>" in res["yanit"]
    assert "</think>" in res["yanit"]
    assert res["ttft_ms"] > 0
    assert res["tpot_ms"] > 0


def test_omni_benchmark_profilleyici():
    """7. OmniBenchmarkProfilleyici 4 amiral gemisi benchmarkı başarıyla tamamlamalıdır."""
    rapor = OmniBenchmarkProfilleyici.calistir_buyuk_final_benchmarki()
    assert len(rapor["gorev_sonuclari"]) == 4
    assert rapor["genel_ortalama_dogruluk"] > 90.0
    assert rapor["triton_flashattention_hizlanma"] >= 3.0


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. OmniGrandFinaleGorsellestirici 6 panelli büyük final teşhis panosunu üretmelidir."""
    cikti = str(tmp_path / "test_grand_finale_paneli.png")
    benchmark_raporu = OmniBenchmarkProfilleyici.calistir_buyuk_final_benchmarki()

    OmniGrandFinaleGorsellestirici.teshis_paneli_olustur(
        benchmark_raporu=benchmark_raporu,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
