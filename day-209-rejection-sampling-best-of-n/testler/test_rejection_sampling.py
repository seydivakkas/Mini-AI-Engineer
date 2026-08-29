"""
PyTest Birim Testleri - Day 209: Rejection Sampling & Best-of-N Motoru.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rejection_sampling_motoru import (
    PolicySampler,
    RejectionFilter,
    RSSFTDatasetBuilder,
    SimplePolicyModel,
    RSSFTTrainer,
)
from src.rejection_profilleyici import RejectionProfilleyici
from src.gorsellestirici import RejectionGorsellestirici


def test_policy_sampler_candidate_count():
    """1. PolicySampler tam olarak K adet aday üretmelidir."""
    adaylar = PolicySampler.orneklem_uret("Soru", k_orneklem=6, sicaklik=0.8)
    assert len(adaylar) == 6
    assert "odul_skoru" in adaylar[0]


def test_policy_sampler_temperature_effect():
    """2. PolicySampler farklı sıcaklıklarda (T=0.2 ve T=1.2) stabil çalışmalıdır."""
    adaylar_low = PolicySampler.orneklem_uret("Soru 1", k_orneklem=4, sicaklik=0.2)
    adaylar_high = PolicySampler.orneklem_uret("Soru 2", k_orneklem=4, sicaklik=1.2)

    assert len(adaylar_low) == 4
    assert len(adaylar_high) == 4


def test_rejection_filter_thresholding():
    """3. RejectionFilter eşik altındaki adayları reddetmelidir."""
    adaylar = [
        {"aday_id": 1, "odul_skoru": 0.40, "yanit": "Yanlış"},
        {"aday_id": 2, "odul_skoru": 0.85, "yanit": "Doğru"},
    ]
    sonuc = RejectionFilter.adaylari_filtrele_ve_sec(adaylar, esik_skoru=0.60)

    assert sonuc["kabul_sayisi"] == 1
    assert sonuc["red_sayisi"] == 1
    assert sonuc["en_iyi_aday"]["aday_id"] == 2
    assert sonuc["en_iyi_kabul_mu"] is True


def test_rejection_filter_empty_exception():
    """4. Boş aday listesi verildiğinde ValueError fırlatılmalıdır."""
    with pytest.raises(ValueError):
        RejectionFilter.adaylari_filtrele_ve_sec([], esik_skoru=0.60)


def test_sft_dataset_builder_structure():
    """5. RSSFTDatasetBuilder doğru veri kümesi formatı üretmelidir."""
    havuz = ["Problem 1", "Problem 2"]
    veri = RSSFTDatasetBuilder.sentetik_veri_seti_olustur(havuz, k_orneklem=4, sicaklik=0.8)

    assert "problem_sayisi" in veri
    assert "sft_veri_kumesi" in veri
    assert veri["problem_sayisi"] == 2


def test_rssft_trainer_step():
    """6. RSSFTTrainer SFT adımı hesaplamalı ve kayıp döndürmelidir."""
    trainer = RSSFTTrainer()
    inp = torch.randint(0, 128, (2, 8))
    tgt = torch.randint(0, 128, (2, 8))
    metrik = trainer.egitim_adimi(inp, tgt)

    assert "sft_loss" in metrik
    assert "perplexity" in metrik
    assert metrik["sft_loss"] > 0.0


def test_profiler_k_scaling_monotonicity():
    """7. Örneklem sayısı K arttıkça en az bir doğru bulma ihtimali artmalıdır."""
    profil = RejectionProfilleyici.profil_raporu_uret()
    pass_oranlari = profil["k_pass_oranlari"]

    assert pass_oranlari[-1] > pass_oranlari[0]
    for i in range(len(pass_oranlari) - 1):
        assert pass_oranlari[i+1] >= pass_oranlari[i]


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. RejectionGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_rejection_paneli.png")
    profil = RejectionProfilleyici.profil_raporu_uret()

    RejectionGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
