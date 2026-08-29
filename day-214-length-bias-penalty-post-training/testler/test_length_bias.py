"""
PyTest Birim Testleri - Day 214: Length-Bias ve Over-Thinking Önleme Motoru.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.length_bias_motoru import (
    LengthPenaltyObjective,
    OverthinkingDetector,
    AdaptiveLengthController,
    LengthRegularizedTrainer,
)
from src.length_bias_profilleyici import LengthBiasProfilleyici
from src.gorsellestirici import LengthBiasGorsellestirici


def test_linear_length_penalty():
    """1. Lineer uzunluk cezası uzun metinlerde ödülü azaltmalıdır."""
    r_kisa = LengthPenaltyObjective.lineer_odul(1.0, 100, alpha=0.001)
    r_uzun = LengthPenaltyObjective.lineer_odul(1.0, 500, alpha=0.001)
    assert r_kisa > r_uzun


def test_hinge_length_penalty_below_target():
    """2. Hedef bütçenin altındaki yanıtlarda ceza 0 olmalıdır."""
    r = LengthPenaltyObjective.hinge_odul(1.0, 300, hedef_uzunluk=500, beta=0.01)
    assert r == 1.0


def test_hinge_length_penalty_above_target():
    """3. Hedef bütçeyi aşan yanıtlarda ceza uygulanmalıdır."""
    r = LengthPenaltyObjective.hinge_odul(1.0, 800, hedef_uzunluk=500, beta=0.01)
    assert r < 1.0


def test_length_normalized_dpo_loss():
    """4. Uzunluk normalize DPO kaybı geçerli tensör üretmelidir."""
    loss = LengthPenaltyObjective.uzunluk_normalize_dpo_kaybi(
        logp_pi_w=torch.tensor(-1.2),
        logp_ref_w=torch.tensor(-2.0),
        logp_pi_l=torch.tensor(-2.5),
        logp_ref_l=torch.tensor(-2.1),
        len_w=150,
        len_l=400,
        beta=0.1,
    )
    assert loss.item() > 0.0


def test_overthinking_detector_detects_loop():
    """5. OverthinkingDetector döngüsel düşünceleri yakalamalıdır."""
    metin = "Dur bir dakika. Tekrar kontrol edeyim. Baştan hesaplayalım."
    analiz = OverthinkingDetector.analiz_et(metin)
    assert analiz["tekrar_sayisi"] >= 2
    assert analiz["gevezelik_skoru"] > 0.30


def test_overthinking_detector_clean_text():
    """6. Temiz metinlerde over-thinking bayrağı konmamalıdır."""
    metin = "5*x = 20 denkleminde her iki tarafı 5'e böleriz ve x=4 buluruz."
    analiz = OverthinkingDetector.analiz_et(metin)
    assert analiz["overthinking_var_mi"] is False


def test_adaptive_budget_allocation():
    """7. Dinamik bütçe karmaşık problemlere daha fazla token vermelidir."""
    b_kolay = AdaptiveLengthController.hedef_butce_belirle("2 + 2 kaç eder?")
    b_zor = AdaptiveLengthController.hedef_butce_belirle("Teoremi ispatlayın.")
    assert b_zor > b_kolay


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. LengthBiasGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_length_bias_paneli.png")
    profil = LengthBiasProfilleyici.verimlilik_profili_cikar()

    LengthBiasGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
