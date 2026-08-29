"""
GÜN 166: GUI Ajanları ve Web Gezintisi Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.set_of_mark_isaretleyici import SetOfMarkIsaretleyici
from src.gui_eylem_uzayi import GUIEylemUzayi
from src.otonom_web_ajani import OtonomWebAjani
from src.gorsellestirici import GUIAjanGorsellestirici


def test_set_of_mark_elemanlari_getir():
    """SoM işaretleyicinin sayfadaki tüm tıklanabilir elemanları getirdiğini test eder."""
    elemanlar = SetOfMarkIsaretleyici.ornek_sayfa_elemanlarini_getir()
    assert len(elemanlar) == 4
    ids = [e["id"] for e in elemanlar]
    assert ids == [1, 2, 3, 4]


def test_set_of_mark_indeksleme():
    """Elemanların ID bazında doğru sözlük yapısına dönüştüğünü test eder."""
    elemanlar = SetOfMarkIsaretleyici.ornek_sayfa_elemanlarini_getir()
    indeks = SetOfMarkIsaretleyici.eleman_etiketle(elemanlar)
    assert 1 in indeks
    assert indeks[1]["etiket"] == "Google Search Input"


def test_gui_eylem_uzayi_click_ayristirma():
    """click(y, x) eyleminin doğru koordinatlarla ayrıştırıldığını test eder."""
    eylem = GUIEylemUzayi.eylem_ayristir("click(345, 500)")
    assert eylem["gecerli_mi"] is True
    assert eylem["tur"] == "click"
    assert eylem["y"] == 345
    assert eylem["x"] == 500


def test_gui_eylem_uzayi_type_ayristirma():
    """type(metin) eyleminin doğru metinle ayrıştırıldığını test eder."""
    eylem = GUIEylemUzayi.eylem_ayristir('type("DeepSeek V3")')
    assert eylem["gecerli_mi"] is True
    assert eylem["tur"] == "type"
    assert eylem["metin"] == "DeepSeek V3"


def test_gui_eylem_uzayi_press_key_scroll_terminate():
    """press_key, scroll ve terminate komutlarının ayrıştırıldığını test eder."""
    k_eylem = GUIEylemUzayi.eylem_ayristir('press_key("Enter")')
    assert k_eylem["tur"] == "press_key"
    assert k_eylem["tus"] == "Enter"

    s_eylem = GUIEylemUzayi.eylem_ayristir('scroll("down")')
    assert s_eylem["tur"] == "scroll"

    t_eylem = GUIEylemUzayi.eylem_ayristir('terminate("SUCCESS")')
    assert t_eylem["tur"] == "terminate"


def test_gui_eylem_uzayi_gecersiz_eylem():
    """Tanımsız komutların geçersiz olarak işaretlendiğini test eder."""
    eylem = GUIEylemUzayi.eylem_ayristir("fly_to_moon(100)")
    assert eylem["gecerli_mi"] is False
    assert eylem["tur"] == "bilinmeyen"


def test_otonom_web_ajani_gorev_yurutme():
    """Ajanın 2 web senaryosunu %100 başarıyla tamamladığını test eder."""
    rapor = OtonomWebAjani.gorevleri_yurut_ve_degerlendir()

    assert rapor["toplam_gorev_sayisi"] == 2
    assert rapor["toplam_adim_sayisi"] == 9
    assert rapor["adim_basari_yuzdesi"] == 100.0
    assert rapor["gorev_tamamlama_orani"] == 100.0


def test_gorsellestirici_pano_uretme():
    """6 panelli GUI ajanı teşhis panosunun PNG olarak kaydedildiğini test eder."""
    rapor = OtonomWebAjani.gorevleri_yurut_ve_degerlendir()
    gorsellestirici = GUIAjanGorsellestirici(dpi=100)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_gui_agent_pano.png")
        gorsellestirici.pano_olustur(rapor, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
