"""
PyTest Birim Testleri - Day 217: SimPO (Simple Preference Optimization) Motoru.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simpo_motoru import (
    SimPORewardCalculator,
    SimPOLossObjective,
    SimPOMemoryProfiler,
    SimPOTrainer,
)
from src.simpo_profilleyici import SimPOProfilleyici
from src.gorsellestirici import SimPOGorsellestirici


def test_simpo_reward_calculator():
    """1. SimPO ödülü log-olasılığı metin uzunluğuna bölmelidir."""
    logp = torch.tensor(-20.0)
    r = SimPORewardCalculator.odul_hesapla(logp, uzunluk_y=10, beta=2.0)
    assert float(r.item()) == -4.0


def test_simpo_loss_margin_enforcement():
    """2. SimPO kaybı hedef marjini (gamma) hesaba katmalıdır."""
    logp_w = torch.tensor(-10.0)
    logp_l = torch.tensor(-30.0)
    kayip, delta_r, _ = SimPOLossObjective.kayip_hesapla(logp_w, logp_l, 10, 10, beta=2.0, gamma_margin=1.0)
    assert delta_r == 4.0
    assert kayip.item() > 0.0


def test_simpo_loss_zero_margin_limit():
    """3. Eşit ödüllerde kayıp pozitif marjin nedeniyle büyür."""
    logp = torch.tensor(-10.0)
    kayip_marjinli, _, _ = SimPOLossObjective.kayip_hesapla(logp, logp, 10, 10, beta=2.0, gamma_margin=1.0)
    kayip_marjinsiz, _, _ = SimPOLossObjective.kayip_hesapla(logp, logp, 10, 10, beta=2.0, gamma_margin=0.0)
    assert kayip_marjinli.item() > kayip_marjinsiz.item()


def test_memory_profiler_savings():
    """4. SimPOMemoryProfiler DPO'ya kıyasla ciddi VRAM tasarrufu raporlamalıdır."""
    tasarruf = SimPOMemoryProfiler.vram_tasarrufu_hesapla(7.0)
    assert tasarruf["simpo_vram_gb"] < tasarruf["dpo_vram_gb"]
    assert tasarruf["tasarruf_yuzde"] > 25.0


def test_simpo_trainer_backward_gradient():
    """5. SimPOTrainer gradyanları başarıyla hesaplamalıdır."""
    adim = SimPOTrainer.egitim_adimi("Soru", "Doğru cevap", "Yanlış cevap")
    assert adim["grad_w"] != 0.0
    assert adim["kayip"] > 0.0


def test_simpo_length_bias_immunity():
    """6. SimPO uzun ve boş şişik yanıtlara uzunluk cezası uygulamalıdır."""
    logp_kisa = torch.tensor(-5.0)  # ortalama -1.0 per token
    logp_uzun_sisik = torch.tensor(-300.0)  # ortalama -3.0 per token
    r_kisa = SimPORewardCalculator.odul_hesapla(logp_kisa, uzunluk_y=5, beta=2.0)
    r_sisik = SimPORewardCalculator.odul_hesapla(logp_uzun_sisik, uzunluk_y=100, beta=2.0)
    assert r_kisa > r_sisik


def test_profiler_margin_analysis():
    """7. Profilleyici gama duyarlılık taramasını eksiksiz içermelidir."""
    prof = SimPOProfilleyici.basarim_profili_cikar()
    assert "marjin_analizi" in prof
    assert len(prof["marjin_analizi"]["gamma_degerleri"]) == 5


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. SimPOGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_simpo_paneli.png")
    profil = SimPOProfilleyici.basarim_profili_cikar()

    SimPOGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
