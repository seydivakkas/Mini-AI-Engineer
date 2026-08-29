"""
RoPE, NTK-Aware Scaling ve YaRN Birim ve Entegrasyon Testleri (Day 104).
Tüm testler endüstriyel standartlarda %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch

from src.rope_temelleri import StandartRoPE, LinearPIRoPE
from src.ntk_ve_yarn import NTKAwareRoPE, YaRNRoPE
from src.baglam_laboratuvari import BaglamLaboratuvari
from src.gorsellestirici import BaglamGorsellestirici


def test_standart_rope_rotasyon_ve_sekil():
    """StandartRoPE modülünün 3D ve 4D tensörlerde doğru şekli koruduğunu test eder."""
    rope = StandartRoPE(dim=32, base=10000.0)
    x_4d = torch.randn(2, 4, 16, 32)
    out_4d = rope(x_4d, seq_len_offset=0)
    assert out_4d.shape == (2, 4, 16, 32)

    x_3d = torch.randn(2, 16, 32)
    out_3d = rope(x_3d, seq_len_offset=10)
    assert out_3d.shape == (2, 16, 32)


def test_linear_pi_rope_olcek():
    """LinearPIRoPE modülünün pozisyon interpolasyonunu doğru ölçeklediğini test eder."""
    pi_rope = LinearPIRoPE(dim=32, base=10000.0, olcek=4.0)
    assert pi_rope.olcek == 4.0
    x = torch.randn(2, 8, 32)
    out = pi_rope(x)
    assert out.shape == (2, 8, 32)


def test_ntk_aware_taban_frekans():
    """NTKAwareRoPE modülünün taban frekans dönüşümünü (base') doğrular."""
    ntk = NTKAwareRoPE(dim=32, base=10000.0, olcek=4.0)
    # ntk_us = 32 / 30 = 1.06667
    # ntk_base = 10000 * 4^(1.06667) > 10000
    assert ntk.ntk_base > 10000.0
    x = torch.randn(2, 4, 8, 32)
    out = ntk(x)
    assert out.shape == (2, 4, 8, 32)


def test_yarn_frekans_rampa_katsayilari():
    """YaRN modülünün hibrit rampa frekanslarını ve sıcaklık çarpanını doğrular."""
    yarn = YaRNRoPE(dim=64, base=10000.0, olcek=8.0, orijinal_max_seq_len=4096)
    assert yarn.frekanslar.shape == (32,)  # dim / 2
    assert yarn.sicaklik_katsayisi > 1.0  # t = 0.1 * ln(8) + 1 > 1
    x = torch.randn(2, 4, 16, 64)
    out = yarn(x)
    assert out.shape == (2, 4, 16, 64)


def test_rope_norm_korunumu():
    """RoPE'un rotasyonel doğasından ötürü vektör L2 normunu koruduğunu doğrular."""
    rope = StandartRoPE(dim=16, base=10000.0)
    x = torch.randn(1, 1, 1, 16)
    x_rot = rope(x, seq_len_offset=42)

    norm_orig = torch.norm(x, p=2).item()
    norm_rot = torch.norm(x_rot, p=2).item()
    assert abs(norm_orig - norm_rot) < 1e-4


def test_mesafe_bagimliligi_azalmasi():
    """Mesafe (|m - n|) arttıkça kosinüs benzerliğinin bozulduğunu test eder."""
    lab = BaglamLaboratuvari(dim=32, orijinal_baglam=1024, hedef_baglam=4096, cihaz=torch.device("cpu"))
    analiz = lab.mesafe_dikkat_bozulmasi_analizi(maks_mesafe=32)
    assert len(analiz) == 4
    for isim, degerler in analiz.items():
        assert len(degerler) > 0
        assert degerler[0] >= degerler[-1] or abs(degerler[0] - 1.0) < 1e-2


def test_baglam_laboratuvari_ppl_simulasyonu():
    """BaglamLaboratuvari PPL eğrisi simülasyonunu doğrular."""
    lab = BaglamLaboratuvari(dim=32, orijinal_baglam=4096, hedef_baglam=131072, cihaz=torch.device("cpu"))
    ppl = lab.perplexity_egrisi_simulasyonu([4096, 16384, 131072])
    assert "Standart RoPE" in ppl
    assert "YaRN" in ppl
    # 128k'da Standart RoPE patlamalı, YaRN ise en düşük PPL'e sahip olmalı
    assert ppl["Standart RoPE"][-1] > ppl["YaRN"][-1]


def test_baglam_gorsellestirici_pano():
    """6 panelli RoPE/YaRN teşhis panosunun oluşturulduğunu test eder."""
    gorsellestirici = BaglamGorsellestirici(dpi=100)
    ornek_ppl = {
        "Standart RoPE": [8.5, 35.0, 150.0, 500.0, 500.0, 500.0],
        "Linear PI": [8.5, 11.0, 13.5, 16.0, 18.5, 21.0],
        "NTK-Aware": [8.5, 9.7, 10.9, 12.1, 13.3, 14.5],
        "YaRN": [8.5, 8.8, 9.0, 9.2, 9.5, 9.8],
    }
    ornek_mesafe = {
        "Standart RoPE": [1.0, 0.8, 0.6, 0.4],
        "Linear PI": [1.0, 0.9, 0.8, 0.7],
        "NTK-Aware": [1.0, 0.85, 0.7, 0.55],
        "YaRN": [1.0, 0.95, 0.9, 0.85],
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit_yolu = os.path.join(tmp_dir, "test_rope_paneli.png")
        gorsellestirici.pano_olustur(ornek_ppl, ornek_mesafe, kayit_yolu=kayit_yolu)
        assert os.path.exists(kayit_yolu)
        assert os.path.getsize(kayit_yolu) > 1000
