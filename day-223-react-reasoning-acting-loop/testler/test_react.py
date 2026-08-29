"""
PyTest Birim Testleri - Day 223: ReAct (Reasoning + Acting) Ajan Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.react_motoru import (
    ReActStep,
    ReActMemoryTrace,
    ReActAgent,
)
from src.react_profilleyici import ReActProfilleyici
from src.gorsellestirici import ReActGorsellestirici


def test_react_step_formatting():
    """1. ReActStep düşünce, eylem ve gözlemi doğru metin formatına dönüştürmelidir."""
    adim = ReActStep(1, "Plan yapıyorum", "Arama[Test]", "Sonuç 123")
    metin = adim.metin_formati()
    assert "Düşünce 1: Plan yapıyorum" in metin
    assert "Eylem 1: Arama[Test]" in metin
    assert "Gözlem 1: Sonuç 123" in metin


def test_react_memory_trace():
    """2. ReActMemoryTrace adımları sırayla hafızaya eklemeli ve tam bağlamı sunmalıdır."""
    hafiza = ReActMemoryTrace()
    hafiza.adim_ekle(ReActStep(1, "D1", "E1", "G1"))
    hafiza.adim_ekle(ReActStep(2, "D2", "E2", "G2"))
    assert len(hafiza.adimlar) == 2
    assert "Düşünce 2: D2" in hafiza.tam_baglam_metni()


def test_action_parser():
    """3. ReActAgent eylem kalıbını (Arac[Arguman]) doğru ayrıştırmalıdır."""
    ajan = ReActAgent()
    arac, arg = ajan.eylem_ayristir("Arama[İstanbul Nüfusu 2024]")
    assert arac == "Arama"
    assert arg == "İstanbul Nüfusu 2024"


def test_tool_execution():
    """4. ReActAgent kayıtlı aracı başarıyla çalıştırıp string gözlem üretmelidir."""
    ajan = ReActAgent()
    ajan.arac_kaydet("KareAl", lambda x: str(int(x) ** 2))
    gozlem = ajan.adim_yurut("KareAl", "9")
    assert gozlem == "81"


def test_unknown_tool_handling():
    """5. ReActAgent tanımlanmamış araç çağrısında hata mesajı dönmelidir."""
    ajan = ReActAgent()
    gozlem = ajan.adim_yurut("OlmayanArac", "Param")
    assert "HATA" in gozlem


def test_finish_action_detection():
    """6. ReActAgent 'Finish' eylemi geldiğinde görevi başarıyla tamamlamalıdır."""
    ajan = ReActAgent()
    plan = [("Sonucu buldum.", "Finish[42]")]
    sonuc = ajan.otonom_coz("Soru", plan)
    assert sonuc["tamamlandi_mi"] is True
    assert sonuc["nihai_cevap"] == "42"


def test_profiler_multihop_gain():
    """7. Profilleyici ReAct mimarisinin çok adımlı doğrulukta %90'ın üzerinde olduğunu raporlamalıdır."""
    prof = ReActProfilleyici.basarim_profili_cikar()
    skor = prof["karsilastirma"]["cok_adimli_dogruluk_yuzdesi"]["ReAct_Mimarisi"]
    assert skor > 90.0


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. ReActGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_react_paneli.png")
    profil = ReActProfilleyici.basarim_profili_cikar()

    ReActGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
