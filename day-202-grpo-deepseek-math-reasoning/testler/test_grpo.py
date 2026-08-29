"""
PyTest Birim Testleri - Day 202: GRPO Matematiksel Akıl Yürütme Motoru.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.grpo_motoru import (
    MathProblemEnvironment,
    RuleBasedMathRewardVerifier,
    PolicyModel,
    GRPOTrainer,
)
from src.grpo_profilleyici import GRPOAkisProfilleyici
from src.gorsellestirici import GRPOGorsellestirici


def test_math_problem_environment():
    """1. MathProblemEnvironment geçerli soru ve doğru cevap üretmelidir."""
    prob = MathProblemEnvironment.rastgele_problem_uret()
    assert "soru" in prob
    assert "dogru_cevap" in prob
    assert len(prob["dogru_cevap"]) > 0


def test_rule_based_verifier_full_reward():
    """2. Biçim ve doğru cevabı içeren yanıt tam puan (1.0) almalıdır."""
    yanit = "<think>Adım adım çözüm yapıldı.</think> Sonuç: 42"
    odul = RuleBasedMathRewardVerifier.odul_hesapla(yanit, "42")
    assert odul["format_odulu"] == 0.2
    assert odul["dogruluk_odulu"] == 0.8
    assert pytest.approx(odul["toplam_odul"]) == 1.0


def test_rule_based_verifier_wrong_answer():
    """3. Yanlış cevaplı yanıt sadece format puanı (0.2) almalıdır."""
    yanit = "<think>Adım adım çözüm yapıldı.</think> Sonuç: 99"
    odul = RuleBasedMathRewardVerifier.odul_hesapla(yanit, "42")
    assert odul["format_odulu"] == 0.2
    assert odul["dogruluk_odulu"] == 0.0
    assert pytest.approx(odul["toplam_odul"]) == 0.2


def test_policy_model_forward():
    """4. PolicyModel doğru çıkış tensör boyutunu üretmelidir."""
    model = PolicyModel(vocab_size=128, embed_dim=64)
    x = torch.randint(0, 128, (2, 8))
    out = model(x)
    assert out.shape == (2, 8, 128)


def test_grpo_advantage_standardization():
    """5. Farklı ödüllere sahip grupta avantajlar sıfır ortalamaya sahip olmalıdır."""
    trainer = GRPOTrainer(group_size=4)
    oduller = [1.0, 0.2, 0.0, 0.0]
    avantajlar = trainer.grup_ici_bagil_avantaj_hesapla(oduller)

    assert avantajlar.shape == (4,)
    assert avantajlar[0] > 0  # En yüksek ödül pozitif avantaj almalı
    assert avantajlar[-1] < 0  # Düşük ödül negatif avantaj almalı


def test_grpo_advantage_zero_variance():
    """6. Gruptaki tüm ödüller eşit olduğunda avantajlar sıfır olmalıdır."""
    trainer = GRPOTrainer(group_size=4)
    oduller = [0.5, 0.5, 0.5, 0.5]
    avantajlar = trainer.grup_ici_bagil_avantaj_hesapla(oduller)

    assert torch.all(avantajlar == 0.0)


def test_grpo_training_step():
    """7. grpo_egitim_adimi geçerli kayıp ve metrikler dönmelidir."""
    trainer = GRPOTrainer(group_size=4)
    prob = {"soru": "x + 2 = 5", "tur": "denklem", "dogru_cevap": "3"}
    res = trainer.grpo_egitim_adimi(prob)

    assert "toplam_kayip" in res
    assert "policy_loss" in res
    assert len(res["grup_ciktilari"]) == 4


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. GRPOGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_grpo_paneli.png")
    profil_raporu = GRPOAkisProfilleyici.egitim_akisini_profili_cikar(adim_sayisi=5)

    GRPOGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
