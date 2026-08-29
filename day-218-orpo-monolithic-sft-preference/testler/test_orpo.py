"""
PyTest Birim Testleri - Day 218: ORPO (Odds Ratio Preference Optimization) Motoru.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.orpo_motoru import (
    SequenceOddsCalculator,
    ORPOLossObjective,
    MonolithicPipelineProfiler,
    ORPOTrainer,
)
from src.orpo_profilleyici import ORPOProfilleyici
from src.gorsellestirici import ORPOGorsellestirici


def test_sequence_odds_calculator():
    """1. SequenceOddsCalculator geçerli olasılık ve odds değeri üretmelidir."""
    logp = torch.tensor(-10.0)
    p, odds = SequenceOddsCalculator.ortalama_olasilik_ve_odds(logp, uzunluk=10)
    assert 0.0 < p.item() < 1.0
    assert odds.item() > 0.0


def test_odds_ratio_positive_for_preferred():
    """2. Tercih edilen yanıtın odds oranı kaybedenden yüksek olmalıdır."""
    odds_w = torch.tensor(2.5)
    odds_l = torch.tensor(0.5)
    log_or = SequenceOddsCalculator.log_odds_ratio(odds_w, odds_l)
    assert log_or.item() > 0.0


def test_orpo_loss_objective_combined():
    """3. ORPO kaybı SFT ve OR kayıplarını doğru ağırlıklandırmalıdır."""
    logp_w = torch.tensor(-5.0)
    logp_l = torch.tensor(-15.0)
    l_orpo, l_sft, l_or, _ = ORPOLossObjective.kayip_hesapla(logp_w, logp_l, 5, 5, lambda_or=0.2)
    assert abs(l_orpo.item() - (l_sft + 0.2 * l_or)) < 1e-4


def test_orpo_loss_backward_gradient():
    """4. ORPO kaybı üzerinden türev başarıyla alınabilmelidir."""
    logp_w = torch.tensor(-6.0, requires_grad=True)
    logp_l = torch.tensor(-18.0, requires_grad=True)
    l_orpo, _, _, _ = ORPOLossObjective.kayip_hesapla(logp_w, logp_l, 5, 5, lambda_or=0.1)
    l_orpo.backward()
    assert logp_w.grad is not None
    assert logp_w.grad.item() != 0.0


def test_monolithic_pipeline_profiler():
    """5. MonolithicPipelineProfiler %40'ın üzerinde GPU saat tasarrufu göstermelidir."""
    prof = MonolithicPipelineProfiler.egitim_sureleri_kiyasla(50000)
    assert prof["orpo_tek_asama_saat"] < prof["sft_dpo_iki_asama_saat"]
    assert prof["tasarruf_yuzde"] > 40.0


def test_orpo_trainer_execution():
    """6. ORPOTrainer tek adım eğitim raporunu eksiksiz üretmelidir."""
    adim = ORPOTrainer.egitim_adimi("Prompt", "Chosen text", "Rejected text", lambda_or=0.1)
    assert "l_orpo_toplam" in adim
    assert "odds_ratio" in adim
    assert adim["grad_w"] != 0.0


def test_profiler_mt_bench_advantage():
    """7. Profilleyici ORPO'nun MT-Bench'te üstün olduğunu raporlamalıdır."""
    prof = ORPOProfilleyici.basarim_profili_cikar()
    mt = prof["karsilastirma"]["mt_bench_skoru"]
    assert mt["Monolitik_ORPO"] > mt["SFT_arti_DPO"]


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. ORPOGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_orpo_paneli.png")
    profil = ORPOProfilleyici.basarim_profili_cikar()

    ORPOGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
