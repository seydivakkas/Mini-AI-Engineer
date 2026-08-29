"""
PyTest Birim Testleri - Day 256: Ses Komutlu Robot Ajanı (Whisper + VLM + VLA).
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.voice_robot_motoru import (
    WhisperSemanticParser,
    VisualSpatialGrounder,
    VoiceConditionedVLAAgent,
)
from src.voice_robot_profilleyici import VoiceRobotProfilleyici
from src.gorsellestirici import VoiceRobotGorsellestirici


def test_whisper_semantic_parser_success():
    """1. parse_command doğal dil komutunu ayrıştırıp alt görevler üretmelidir."""
    res = WhisperSemanticParser.parse_command("Kırmızı kupayı su ısıtıcısının yanına koy")
    assert res["niyet"] == "AL_VE_TASI"
    assert res["belirsizlik_var_mi"] is False
    assert res["hedef_nesne"] == "kırmızı kupa"
    assert len(res["alt_gorevler"]) == 5


def test_whisper_semantic_parser_ambiguity():
    """2. parse_command belirsiz komutta netleştirme sorusu üretmelidir."""
    res = WhisperSemanticParser.parse_command("Bardağı getir")
    assert res["belirsizlik_var_mi"] is True
    assert "netlestirme_sorusu" in res


def test_visual_spatial_grounder_known_object():
    """3. ground_object bilinen nesneyi 3D koordinatlara bağlamalıdır."""
    res = VisualSpatialGrounder.ground_object("kırmızı kupa")
    assert res["bulundu"] is True
    assert len(res["koordinat_3d_m"]) == 3
    assert res["vlm_guven_skoru"] >= 0.95


def test_visual_spatial_grounder_unknown_object():
    """4. ground_object sahnede olmayan nesne için bulundu=False dönmelidir."""
    res = VisualSpatialGrounder.ground_object("altın anahtar")
    assert res["bulundu"] is False


def test_voice_vla_agent_init():
    """5. VoiceConditionedVLAAgent başlangıç durumunu doğru kurmalıdır."""
    agent = VoiceConditionedVLAAgent()
    assert agent.durum == "BEKLEMEDE"
    assert agent.maks_hiz_m_s == 0.25


def test_voice_vla_agent_full_pipeline():
    """6. process_voice_instruction uçtan uca görev planı ve sesli yanıt üretmelidir."""
    agent = VoiceConditionedVLAAgent()
    res = agent.process_voice_instruction("Kırmızı kupayı su ısıtıcısına taşı")
    assert res["basarili"] is True
    assert res["durum"] == "GOREV_TAMAMLANDI"
    assert len(res["icra_plani"]) == 5
    assert "kırmızı kupa" in res["sesli_geri_bildirim"]


def test_voice_robot_profiler_output():
    """7. VoiceRobotProfilleyici kıyaslama metriklerini eksiksiz üretmelidir."""
    profil = VoiceRobotProfilleyici.basarim_profili_cikar()
    assert "Whisper_VLM_VLA" in profil["karsilastirma"]["dogal_ses_komut_anlama_yuzde"]
    assert profil["karsilastirma"]["dogal_ses_komut_anlama_yuzde"]["Whisper_VLM_VLA"] == 98.4


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. VoiceRobotGorsellestirici 6 panelli teşhis panosunu üretmelidir."""
    cikti = str(tmp_path / "test_voice_robot_paneli.png")
    profil = VoiceRobotProfilleyici.basarim_profili_cikar()

    VoiceRobotGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
