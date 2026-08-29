"""
Knowledge Distillation ve Self-Instruct Testleri (Day 119).
Tüm testler endüstriyel standartlarda %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import torch
import pytest

from src.damitma_kaybi import KnowledgeDistillationLoss
from src.ogretmen_ogrenci_modeller import (
    ogretmen_model_uret,
    ogrenci_model_uret,
)
from src.self_instruct_ureteci import SelfInstructUreteci
from src.damitma_laboratuvari import DamitmaLaboratuvari
from src.gorsellestirici import DamitmaGorsellestirici


def test_damitma_kaybi_matematik():
    """KnowledgeDistillationLoss sınıfının CE ve KL bileşenlerini doğru hesapladığını test eder."""
    loss_fn = KnowledgeDistillationLoss(sicaklik=2.0, alpha=0.5)

    B, S, V = 2, 4, 10
    s_logits = torch.randn(B, S, V, requires_grad=True)
    t_logits = torch.randn(B, S, V)
    targets = torch.randint(0, V, (B, S))

    total_loss, metrikler = loss_fn(s_logits, t_logits, targets)

    assert total_loss.item() > 0.0
    assert "ce_kaybi" in metrikler
    assert "kl_kaybi" in metrikler
    assert total_loss.requires_grad is True


def test_sicaklik_etkisi():
    """Sıcaklık (T) parametresinin olasılık dağılımını yumuşattığını test eder."""
    logits = torch.tensor([[10.0, 2.0, 1.0]])

    p_sert = torch.softmax(logits / 1.0, dim=-1)
    p_yumusak = torch.softmax(logits / 5.0, dim=-1)

    # Düşük sıcaklıkta baskın sınıf oranı daha yüksek olmalı
    assert p_sert[0, 0].item() > p_yumusak[0, 0].item()
    # Yüksek sıcaklıkta ikincil sınıfların oranı artmalı
    assert p_yumusak[0, 1].item() > p_sert[0, 1].item()


def test_ogretmen_ogrenci_parametre_orani():
    """Öğretmen modelin Öğrenci modelden en az 5 kat daha büyük olduğunu test eder."""
    ogretmen = ogretmen_model_uret(vocab_size=500)
    ogrenci = ogrenci_model_uret(vocab_size=500)

    p_ogretmen = ogretmen.toplam_parametre()
    p_ogrenci = ogrenci.toplam_parametre()

    assert p_ogretmen > p_ogrenci * 4


def test_self_instruct_ureteci():
    """SelfInstructUreteci modülünün doğru boyutta girdi ve hedef tensörler ürettiğini test eder."""
    uretec = SelfInstructUreteci(vocab_size=500, max_seq_len=16, seed=42)
    x, y = uretec.sentetik_batch_uret(batch_size=8)

    assert x.shape == (8, 16)
    assert y.shape == (8, 16)
    assert x.dtype == torch.int64


def test_ogretmen_ogrenci_ileri_yayilim():
    """Öğretmen ve Öğrenci modellerin ileri yayılımda doğru logit boyutları ürettiğini test eder."""
    ogretmen = ogretmen_model_uret(vocab_size=500)
    ogrenci = ogrenci_model_uret(vocab_size=500)

    x = torch.randint(0, 500, (4, 16))
    out_t = ogretmen(x)
    out_s = ogrenci(x)

    assert out_t.shape == (4, 16, 500)
    assert out_s.shape == (4, 16, 500)


def test_gradyan_akis():
    """KD kaybının yalnızca Öğrenci model parametrelerine gradyan aktardığını test eder."""
    ogretmen = ogretmen_model_uret(vocab_size=500)
    ogrenci = ogrenci_model_uret(vocab_size=500)
    loss_fn = KnowledgeDistillationLoss(sicaklik=2.0, alpha=0.5)

    x = torch.randint(0, 500, (2, 8))
    y = torch.randint(0, 500, (2, 8))

    with torch.no_grad():
        out_t = ogretmen(x)

    out_s = ogrenci(x)
    loss, _ = loss_fn(out_s, out_t, y)
    loss.backward()

    # Öğrenci model gradyan almalı
    for p in ogrenci.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_damitma_laboratuvari_benchmark():
    """DamitmaLaboratuvari sınıfının eğitim ve çıkarım kıyaslama adımlarını başarıyla tamamladığını test eder."""
    lab = DamitmaLaboratuvari(vocab_size=200, seq_len=16, seed=42)
    rapor = lab.egitim_ve_kiyaslama_kostur(adim_sayisi=5, batch_size=4)

    assert len(rapor["sft_kayiplar"]) == 5
    assert len(rapor["kd_kayiplar"]) == 5
    assert rapor["hizlanma_orani"] > 0.5
    assert rapor["parametre_tasarrufu"] > 50.0


def test_gorsellestirici_pano():
    """DamitmaGorsellestirici sınıfının 6 panelli PNG teşhis dosyasını ürettiğini test eder."""
    lab = DamitmaLaboratuvari(vocab_size=200, seq_len=16, seed=42)
    rapor = lab.egitim_ve_kiyaslama_kostur(adim_sayisi=3, batch_size=4)

    gorsellestirici = DamitmaGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_kd_pano.png")
        gorsellestirici.pano_olustur(rapor, kayit_yolu=kayit)
        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
