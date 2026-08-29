"""
PyTest Birim Testleri - Day 212: Constitutional AI (CAI) Motoru.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.constitutional_motoru import (
    Constitution,
    SelfCritiqueEngine,
    RevisionEngine,
    RLAIFFeedbackModel,
    CAIPostTrainer,
)
from src.constitutional_profilleyici import ConstitutionalProfilleyici
from src.gorsellestirici import ConstitutionalGorsellestirici


def test_constitution_principles():
    """1. Anayasal ilkeler eksiksiz yüklenmelidir."""
    assert "C1_ZARARSIZLIK" in Constitution.ILKELER
    assert "C3_ASIRI_REDDETMEME" in Constitution.ILKELER


def test_self_critique_harm_detection():
    """2. SelfCritiqueEngine zararlı içerikleri tespit edip ihlal bayrağı koymalıdır."""
    rapor = SelfCritiqueEngine.elestiri_yap(
        prompt="Saldırı yap",
        ham_yanit="Bu virüs ile sisteme saldırı yapabilirsiniz.",
        ilke_kodu="C1_ZARARSIZLIK",
    )
    assert rapor["ihlal_var_mi"] is True
    assert "Eleştiri" in rapor["elestiri_metni"]


def test_self_critique_safe_response():
    """3. Güvenli yanıtlarda ihlal bayrağı konmamalıdır."""
    rapor = SelfCritiqueEngine.elestiri_yap(
        prompt="Python for döngüsü nasıl yazılır?",
        ham_yanit="for i in range(10): print(i)",
        ilke_kodu="C1_ZARARSIZLIK",
    )
    assert rapor["ihlal_var_mi"] is False


def test_revision_engine_rewrite():
    """4. RevisionEngine zararlı yanıtı güvenli formata dönüştürmelidir."""
    elestiri = {
        "ilke_kodu": "C1_ZARARSIZLIK",
        "ilke_adi": "Zararsızlık",
        "ihlal_var_mi": True,
    }
    duzeltilmis = RevisionEngine.duzeltme_yap("Soru", "Zararlı kod", elestiri)
    assert "şifreleme algoritmaları" in duzeltilmis


def test_revision_engine_no_change_on_safe():
    """5. İhlal olmayan yanıtlarda metin değiştirilmeden döndürülmelidir."""
    elestiri = {"ilke_kodu": "C1_ZARARSIZLIK", "ilke_adi": "Zararsızlık", "ihlal_var_mi": False}
    orijinal = "Güvenli yanıt metni"
    sonuc = RevisionEngine.duzeltme_yap("Soru", orijinal, elestiri)
    assert sonuc == orijinal


def test_rlaif_feedback_model():
    """6. RLAIFFeedbackModel güvenli yanıtı zararlı yanıta tercih etmelidir."""
    tercih = RLAIFFeedbackModel.tercih_belirle(
        prompt="Test",
        yanit_a="Güvenli savunma tavsiyesi",
        yanit_b="Sisteme saldırı ve virüs yükleme",
    )
    assert tercih["kazanan"] == "A"
    assert tercih["tercih_olasiligi_A"] > 0.50


def test_cai_post_trainer_pipeline():
    """7. CAIPostTrainer tam CAI akışını eksiksiz yürütmelidir."""
    sonuc = CAIPostTrainer.anayasal_hizalama_adimi(
        prompt="Saldırı yap",
        ham_yanit="Saldırı yöntemi: virüs",
        ilke_kodu="C1_ZARARSIZLIK",
    )
    assert "elestiri" in sonuc
    assert "duzeltilmis_yanit" in sonuc
    assert "rlaif_degerlendirme" in sonuc


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. ConstitutionalGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_cai_paneli.png")
    profil = ConstitutionalProfilleyici.guvenlik_profili_cikar()

    ConstitutionalGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
