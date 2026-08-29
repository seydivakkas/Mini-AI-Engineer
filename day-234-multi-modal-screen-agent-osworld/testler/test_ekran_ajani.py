"""
PyTest Birim Testleri - Day 234: Çok Modlu Ekran Ajanı (Computer Use / OSWorld) Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ekran_ajani_motoru import (
    ScreenElement,
    GUIAction,
    ComputerUseAgent,
)
from src.osworld_profilleyici import OSWorldProfilleyici
from src.gorsellestirici import EkranGorsellestirici


def test_screen_element_center_coordinates():
    """1. ScreenElement merkez koordinatını (x_c, y_c) doğru hesaplamalıdır."""
    el = ScreenElement(1, "Button", 100, 200, 50, 30)
    assert el.merkez_koordinati() == (125, 215)


def test_gui_action_formatting():
    """2. GUIAction eylem türüne göre doğru formatlama metni üretmelidir."""
    a1 = GUIAction("CLICK", x=100, y=200)
    a2 = GUIAction("TYPE", metin="test")
    a3 = GUIAction("HOTKEY", tuslar=["Ctrl", "C"])
    assert "CLICK(x=100, y=200)" in a1.format_metni()
    assert "TYPE(text='test')" in a2.format_metni()
    assert "HOTKEY(Ctrl+C)" in a3.format_metni()


def test_agent_register_and_find_element():
    """3. ComputerUseAgent ekrana bileşen ekleyip etiketle bulabilmelidir."""
    agent = ComputerUseAgent()
    agent.ekrana_bilesen_ekle(ScreenElement(1, "Hesap_Makinesi", 50, 50, 40, 40))
    el = agent.bilesen_bul("Hesap_Makinesi")
    assert el is not None
    assert el.x == 50


def test_agent_click_execution():
    """4. ComputerUseAgent tıklama eyleminde fare konumunu güncellemelidir."""
    agent = ComputerUseAgent()
    sonuc = agent.eylem_icra_et(GUIAction("CLICK", x=300, y=400))
    assert sonuc["durum"] == "BASARILI"
    assert agent.fare_konumu == (300, 400)


def test_agent_type_execution():
    """5. ComputerUseAgent metin yazma eylemini başarıyla kaydetmelidir."""
    agent = ComputerUseAgent()
    sonuc = agent.eylem_icra_et(GUIAction("TYPE", metin="Hello OS"))
    assert sonuc["durum"] == "BASARILI"
    assert "Hello OS" in sonuc["log"]


def test_agent_hotkey_execution():
    """6. ComputerUseAgent kısayol tuşu eylemini başarıyla icra etmelidir."""
    agent = ComputerUseAgent()
    sonuc = agent.eylem_icra_et(GUIAction("HOTKEY", tuslar=["Alt", "F4"]))
    assert sonuc["durum"] == "BASARILI"
    assert "Alt+F4" in sonuc["log"]


def test_profiler_osworld_metrics():
    """7. Profilleyici Ekran Ajanı OSWorld başarısının %80 üzerinde olduğunu doğrulamalıdır."""
    prof = OSWorldProfilleyici.basarim_profili_cikar()
    skor = prof["karsilastirma"]["osworld_gorev_basarisi"]["Cok_Modlu_Ekran_Ajani"]
    assert skor > 80.0


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. EkranGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_ekran_paneli.png")
    profil = OSWorldProfilleyici.basarim_profili_cikar()

    EkranGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
