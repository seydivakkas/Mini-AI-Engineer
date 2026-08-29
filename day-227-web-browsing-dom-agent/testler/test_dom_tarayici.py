"""
PyTest Birim Testleri - Day 227: Web Tarayıcı ve DOM Ağacı Ajan Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dom_tarayici_motoru import (
    DOMElement,
    DOMTreePruner,
    WebBrowsingAgent,
)
from src.dom_profilleyici import DOMProfilleyici
from src.gorsellestirici import DOMGorsellestirici


def test_dom_element_formatting():
    """1. DOMElement nesnesi etkileşimli elemanları [ID] formatında biçimlendirmelidir."""
    el = DOMElement(1, "button", "Gönder", {"type": "submit"}, etkilesimli_mi=True)
    metin = el.format_metni()
    assert "[1] <button" in metin
    assert "Gönder" in metin


def test_html_pruning_script_removal():
    """2. DOMTreePruner HTML içindeki script ve style etiketlerini temizlemelidir."""
    html = "<div><script>alert(1);</script><p>Temiz Metin</p></div>"
    elemanlar = DOMTreePruner.html_temizle_ve_buda(html)
    tum_metin = DOMTreePruner.agaci_metne_donustur(elemanlar)
    assert "alert" not in tum_metin
    assert "Temiz Metin" in tum_metin


def test_interactive_id_assignment():
    """3. DOMTreePruner buton ve giriş kutularına sıralı ID atamalıdır."""
    html = "<input placeholder='İsim' /><button>Tıkla</button>"
    elemanlar = DOMTreePruner.html_temizle_ve_buda(html)
    ids = [e.eleman_id for e in elemanlar if e.etkilesimli_mi]
    assert 1 in ids
    assert 2 in ids


def test_agent_click_action():
    """4. WebBrowsingAgent geçerli bir ID için tıklama eylemini başarıyla icra etmelidir."""
    html = "<button>Satın Al</button>"
    ajan = WebBrowsingAgent(baslangic_html=html)
    sonuc = ajan.eylem_icra_et("Click[1]")
    assert "GÖZLEM" in sonuc
    assert "tıklandı" in sonuc


def test_agent_type_action():
    """5. WebBrowsingAgent giriş kutusuna metin yazma eylemini icra etmelidir."""
    html = "<input type='text' />"
    ajan = WebBrowsingAgent(baslangic_html=html)
    sonuc = ajan.eylem_icra_et("Type[1, 'Telefon']")
    assert "Telefon" in sonuc
    assert "yazıldı" in sonuc


def test_agent_finish_action():
    """6. WebBrowsingAgent Finish eylemiyle görevi tamamlamalıdır."""
    html = "<div>Sayfa</div>"
    ajan = WebBrowsingAgent(baslangic_html=html)
    sonuc = ajan.otonom_gezin("Hedef", ["Finish['Sonuç Bulundu']"])
    assert sonuc["tamamlandi_mi"] is True
    assert sonuc["nihai_yanit"] == "Sonuç Bulundu"


def test_profiler_dom_metrics():
    """7. Profilleyici budanmış DOM ağacının başarı oranının %90 üstünde olduğunu doğrulamalıdır."""
    prof = DOMProfilleyici.basarim_profili_cikar()
    skor = prof["karsilastirma"]["web_navigasyon_basarisi"]["Budanmis_DOM_Agaci"]
    assert skor > 90.0


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. DOMGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_dom_paneli.png")
    profil = DOMProfilleyici.basarim_profili_cikar()

    DOMGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
