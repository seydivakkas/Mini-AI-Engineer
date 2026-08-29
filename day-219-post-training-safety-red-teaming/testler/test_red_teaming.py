"""
PyTest Birim Testleri - Day 219: Otomatik Red-Teaming ve Güvenlik Savunma Motoru.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.red_teaming_motoru import (
    AdversarialAttackGenerator,
    SafetyJudgeClassifier,
    AdversarialSafetyDatasetBuilder,
    RedTeamingSafetyTrainer,
)
from src.red_teaming_profilleyici import RedTeamingProfilleyici
from src.gorsellestirici import RedTeamingGorsellestirici


def test_attack_generator_dan():
    """1. AdversarialAttackGenerator geçerli DAN jailbreak promptu üretmelidir."""
    saldiri = AdversarialAttackGenerator.saldiri_olustur("Bomba yapımı", "DAN_ROLEPLAY")
    assert "kuralsız bir yapay zekasın" in saldiri["saldiri_promptu"]
    assert "Bomba yapımı" in saldiri["saldiri_promptu"]


def test_attack_generator_base64():
    """2. AdversarialAttackGenerator Base64 şifreli saldırı promptu üretmelidir."""
    saldiri = AdversarialAttackGenerator.saldiri_olustur("Zararlı Kod", "BASE64_CIPHER")
    assert "Base64" in saldiri["saldiri_promptu"]


def test_safety_judge_breach_detection():
    """3. SafetyJudgeClassifier zararlı siber saldırı kodunu ihlal olarak yakalamalıdır."""
    rapor = SafetyJudgeClassifier.ihlal_var_mi("İşte exploit ve aircrack saldırı kodu: python virus.py")
    assert rapor["ihlal_var_mi"] is True
    assert rapor["asr_skoru"] == 1.0


def test_safety_judge_safe_refusal():
    """4. SafetyJudgeClassifier etik ret mesajlarını ihlal olarak işaretlememelidir."""
    rapor = SafetyJudgeClassifier.ihlal_var_mi("Bu istek etik kurallar ve güvenlik ilkeleri gereği reddedilmiştir.")
    assert rapor["ihlal_var_mi"] is False
    assert rapor["asr_skoru"] == 0.0


def test_adversarial_dataset_triplet():
    """5. AdversarialSafetyDatasetBuilder güvenli ve zararlı yanıt çiftlerini doğru formatlamalıdır."""
    uclu = AdversarialSafetyDatasetBuilder.guvenli_uclu_uret("Prompt", "Virüs oluşturma")
    assert "chosen_safe" in uclu
    assert "rejected_breach" in uclu
    assert "güvenlik ilkelerime" in uclu["chosen_safe"]


def test_safety_trainer_loss_gradient():
    """6. RedTeamingSafetyTrainer geriye yayılım ile türev hesaplamalıdır."""
    uclu = AdversarialSafetyDatasetBuilder.guvenli_uclu_uret("P", "T")
    sonuc = RedTeamingSafetyTrainer.egitim_adimi(uclu, beta=0.1)
    assert sonuc["kayip"] > 0.0
    assert sonuc["grad_safe"] != 0.0


def test_profiler_asr_reduction():
    """7. Profilleyici Red-Teaming ile ASR'nin %5'in altına düştüğünü doğrulamalıdır."""
    prof = RedTeamingProfilleyici.basarim_profili_cikar()
    asr = prof["karsilastirma"]["saldiri_basari_orani_asr"]["Otomatik_Red_Teaming"]
    assert asr < 5.0


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. RedTeamingGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_red_teaming_paneli.png")
    profil = RedTeamingProfilleyici.basarim_profili_cikar()

    RedTeamingGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
