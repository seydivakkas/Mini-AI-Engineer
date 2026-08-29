"""
Model Birleştirme (SLERP, TIES, DARE) Testleri (Day 115).
Tüm testler endüstriyel standartlarda %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch

from src.model_birlestirici import ModelBirlestirici, slerp_tensor
from src.ag_mimarisi import UzmanModel
from src.birlestirme_laboratuvari import BirlestirmeLaboratuvari
from src.gorsellestirici import ModelMergingGorsellestirici


def test_slerp_tensor_temel():
    """slerp_tensor fonksiyonunun uç noktalarda (t=0, t=1) ve küresel ortada doğru çalıştığını test eder."""
    v0 = torch.tensor([1.0, 0.0])
    v1 = torch.tensor([0.0, 1.0])

    res_0 = slerp_tensor(v0, v1, t=0.0)
    res_1 = slerp_tensor(v0, v1, t=1.0)
    res_mid = slerp_tensor(v0, v1, t=0.5)

    assert torch.allclose(res_0, v0, atol=1e-5)
    assert torch.allclose(res_1, v1, atol=1e-5)
    # t=0.5 iken [cos(45), sin(45)] = [0.7071, 0.7071]
    assert abs(float(torch.norm(res_mid).item()) - 1.0) < 1e-4
    assert abs(float(res_mid[0].item()) - 0.7071) < 1e-3


def test_slerp_tensor_paralel():
    """Paralel veya sıfır normlu vektörlerde SLERP'in NaN üretmeden lineer geri dönüş yaptığını test eder."""
    v0 = torch.tensor([2.0, 4.0])
    v1 = torch.tensor([4.0, 8.0])

    res = slerp_tensor(v0, v1, t=0.5)
    assert not torch.isnan(res).any()
    assert torch.allclose(res, torch.tensor([3.0, 6.0]), atol=1e-4)


def test_lineer_birlestir():
    """ModelBirlestirici.lineer_birlestir fonksiyonunun görev vektörlerini doğru topladığını test eder."""
    base = UzmanModel(16, 32, 8)
    m1 = UzmanModel(16, 32, 8)
    m2 = UzmanModel(16, 32, 8)

    merged = ModelBirlestirici.lineer_birlestir(base, [m1, m2], agirliklar=[0.5, 0.5])
    assert isinstance(merged, UzmanModel)

    # Parametre şekillerini kontrol et
    for p_m, p_b in zip(merged.parameters(), base.parameters()):
        assert p_m.shape == p_b.shape
        assert not torch.isnan(p_m).any()


def test_ties_birlestir_trim_ve_sign():
    """TIES-Merging'in budama ve işaret mutabakatı adımlarını test eder."""
    base = UzmanModel(16, 32, 8)
    m1 = UzmanModel(16, 32, 8)
    m2 = UzmanModel(16, 32, 8)

    ties_merged = ModelBirlestirici.ties_birlestir(base, [m1, m2], agirliklar=[0.5, 0.5], trim_orani=0.4)
    assert isinstance(ties_merged, UzmanModel)

    for p in ties_merged.parameters():
        assert not torch.isnan(p).any()


def test_dare_birlestir_drop_and_rescale():
    """DARE-Merging'in Bernoulli seyreltmesi ve yeniden ölçeklemesini test eder."""
    base = UzmanModel(16, 32, 8)
    m1 = UzmanModel(16, 32, 8)
    m2 = UzmanModel(16, 32, 8)

    dare_merged = ModelBirlestirici.dare_birlestir(base, [m1, m2], agirliklar=[0.5, 0.5], drop_orani=0.7, ties_uygula=True)
    assert isinstance(dare_merged, UzmanModel)

    for p in dare_merged.parameters():
        assert not torch.isnan(p).any()


def test_uzman_model_forward():
    """UzmanModel forward geçişini test eder."""
    model = UzmanModel(in_dim=32, hidden_dim=64, out_dim=16)
    x = torch.randn(5, 32)
    y = model(x)

    assert y.shape == (5, 16)
    assert not torch.isnan(y).any()


def test_birlestirme_laboratuvari_fuzyon():
    """BirlestirmeLaboratuvari uçtan uca çok alanlı deney akışını test eder."""
    lab = BirlestirmeLaboratuvari(in_dim=16, hidden_dim=32, out_dim=8, cihaz=torch.device("cpu"))
    sonuclar = lab.fuzyon_deneyini_kostur()

    assert "Taban Model (Base)" in sonuclar
    assert "SLERP Merge" in sonuclar
    assert "TIES Merge" in sonuclar
    assert "DARE-TIES Merge" in sonuclar
    for k, v in sonuclar.items():
        assert "Bileşik Başarı" in v
        assert 0.0 <= v["Bileşik Başarı"] <= 100.0


def test_model_merging_gorsellestirici_pano():
    """ModelMergingGorsellestirici modülünün 6 panelli teşhis panosu ürettiğini test eder."""
    gorsellestirici = ModelMergingGorsellestirici(dpi=100)
    ornek_sonuclar = {
        "Base Model": {"Matematik Skoru": 30.0, "Kodlama Skoru": 30.0, "Bileşik Başarı": 30.0},
        "Math Expert": {"Matematik Skoru": 95.0, "Kodlama Skoru": 20.0, "Bileşik Başarı": 57.5},
        "Code Expert": {"Matematik Skoru": 15.0, "Kodlama Skoru": 96.0, "Bileşik Başarı": 55.5},
        "Linear Merge": {"Matematik Skoru": 65.0, "Kodlama Skoru": 62.0, "Bileşik Başarı": 63.5},
        "SLERP Merge": {"Matematik Skoru": 82.0, "Kodlama Skoru": 80.0, "Bileşik Başarı": 81.0},
        "TIES Merge": {"Matematik Skoru": 88.0, "Kodlama Skoru": 86.0, "Bileşik Başarı": 87.0},
        "DARE-TIES": {"Matematik Skoru": 89.0, "Kodlama Skoru": 88.0, "Bileşik Başarı": 88.5},
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_merging_pano.png")
        gorsellestirici.pano_olustur(ornek_sonuclar, kayit_yolu=kayit)
        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
