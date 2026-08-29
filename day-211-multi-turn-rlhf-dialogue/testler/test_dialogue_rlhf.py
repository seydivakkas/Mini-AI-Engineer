"""
PyTest Birim Testleri - Day 211: Çok Turlu Diyalog RLHF Motoru.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dialogue_rlhf_motoru import (
    DialogueState,
    UserSimulator,
    MultiTurnRewardModel,
    TemporalCreditAssigner,
    MultiTurnRLHFTrainer,
)
from src.dialogue_profilleyici import DialogueProfilleyici
from src.gorsellestirici import DialogueGorsellestirici


def test_dialogue_state_tracking():
    """1. DialogueState konuşma geçmişini doğru saklamalıdır."""
    state = DialogueState(sistem_mesaji="Test Asistanı")
    state.tur_ekle("user", "Merhaba")
    state.tur_ekle("assistant", "Selam, nasıl yardımcı olabilirim?")

    assert state.tur_sayisi() == 2
    assert "[USER]: Merhaba" in state.tam_baglami_getir()


def test_user_simulator_dialogue_flow():
    """2. UserSimulator diyalog adımlarını sırayla ilerletmelidir."""
    user = UserSimulator()
    m1 = user.sonraki_kullanici_mesaji()
    m2 = user.sonraki_kullanici_mesaji()

    assert m1 is not None
    assert m2 is not None
    assert m1 != m2


def test_reward_model_step_reward():
    """3. MultiTurnRewardModel geçerli ara adım ödülü üretmelidir."""
    r = MultiTurnRewardModel.ara_adim_odulu(
        gecmis_baglam="Önceki konuşma",
        yeni_yanit="B-Tree index ekleyerek sorgu optimizasyonu yapabilirsiniz.",
        kullanici_sorusu="Sorgum yavaş",
    )
    assert isinstance(r, float)
    assert r > 0.0


def test_reward_model_repetition_penalty():
    """4. Tekrara düşüldüğünde ceza uygulanmalıdır."""
    baglam = "Bu bir test cevabıdır."
    r = MultiTurnRewardModel.ara_adim_odulu(
        gecmis_baglam=baglam,
        yeni_yanit=baglam,
        kullanici_sorusu="Soru",
    )
    assert r < 0.0


def test_temporal_credit_assignment():
    """5. TemporalCreditAssigner geriye dönük indirimli getiriyi doğru hesaplamalıdır."""
    oduller = [1.0, 1.0, 2.0]
    getiriler = TemporalCreditAssigner.birikimli_getiri_hesapla(oduller, gamma=0.5)

    # G_2 = 2.0
    # G_1 = 1.0 + 0.5 * 2.0 = 2.0
    # G_0 = 1.0 + 0.5 * 2.0 = 2.0
    assert getiriler[2] == 2.0
    assert getiriler[1] == 2.0
    assert getiriler[0] == 2.0


def test_trainer_dialogue_execution():
    """6. MultiTurnRLHFTrainer tüm epizodu başarıyla yürütmelidir."""
    trainer = MultiTurnRLHFTrainer(gamma=0.95)
    sonuc = trainer.tam_diyalog_yurut()

    assert sonuc["toplam_tur"] > 0
    assert "indirimli_getiriler" in sonuc
    assert len(sonuc["indirimli_getiriler"]) == sonuc["toplam_tur"]


def test_profiler_output_structure():
    """7. DialogueProfilleyici metrikleri eksiksiz üretmelidir."""
    profil = DialogueProfilleyici.profil_raporu_uret()
    assert "karsilastirma" in profil
    assert "tur_odulleri" in profil


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. DialogueGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_dialogue_paneli.png")
    profil = DialogueProfilleyici.profil_raporu_uret()

    DialogueGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
