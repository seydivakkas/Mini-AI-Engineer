"""
GÜN 153: Tümdengelimsel Mantık ve Safsata Dedektörü Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.oncul_sonuc_ayristirici import OnculSonucAyristirici
from src.safsata_tespitcisi import SafsataTespitcisi
from src.tumdengelim_motoru import TumdengelimMotoru
from src.gorsellestirici import MantikGorsellestirici


def test_oncul_sonuc_ayristirici():
    """Öncül ve sonuç ayrıştırıcısının cümleleri doğru ayırdığını test eder."""
    metin = "Tüm insanlar ölümlüdür. Sokrates bir insandır. Dolayısıyla Sokrates ölümlüdür."
    ayristirma = OnculSonucAyristirici.ayristir(metin)

    assert len(ayristirma["onculler"]) == 2
    assert "Sokrates ölümlüdür" in ayristirma["sonuc"]


def test_safsata_tespitcisi_ad_hominem():
    """Kişiye saldırı (Ad Hominem) safsatasının doğru tespit edildiğini test eder."""
    onculler = ["Sen zaten eğitimsizsin ve diplomasız birisin."]
    sonuc = "Bu yüzden senin ekonomiye dair argümanın tamamen yanlıştır."

    rapor = SafsataTespitcisi.safsata_tara(onculler, sonuc)
    assert rapor["safsata_var_mi"] is True
    assert rapor["safsata_anahtari"] == "ad_hominem"


def test_safsata_tespitcisi_straw_man():
    """Korkuluk / Çarpıtma (Straw Man) safsatasının doğru tespit edildiğini test eder."""
    onculler = ["Karşı taraf yapay zekanın etik denetimden geçmesini istiyor."]
    sonuc = "Demek ki onlar bütün teknolojiyi çöpe atıp tamamen yasaklamak istiyor."

    rapor = SafsataTespitcisi.safsata_tara(onculler, sonuc)
    assert rapor["safsata_var_mi"] is True
    assert rapor["safsata_anahtari"] == "straw_man"


def test_safsata_tespitcisi_false_dilemma():
    """Yanlış İkilem (False Dilemma) safsatasının doğru tespit edildiğini test eder."""
    onculler = ["Bu projeye tam destek vermiyorsun."]
    sonuc = "O halde ya bizimlesin ya da projenin başarısızlığını isteyen bir düşmansın."

    rapor = SafsataTespitcisi.safsata_tara(onculler, sonuc)
    assert rapor["safsata_var_mi"] is True
    assert rapor["safsata_anahtari"] == "false_dilemma"


def test_safsata_tespitcisi_circular_reasoning():
    """Kısır Döngü (Circular Reasoning) safsatasının doğru tespit edildiğini test eder."""
    onculler = ["Bu rapor tamamen hatasızdır."]
    sonuc = "Çünkü kendisi öyle diyor ve içinde asla yanlış olmayacağı yazıyor."

    rapor = SafsataTespitcisi.safsata_tara(onculler, sonuc)
    assert rapor["safsata_var_mi"] is True
    assert rapor["safsata_anahtari"] == "circular_reasoning"


def test_safsata_tespitcisi_affirming_consequent():
    """Sonucun Doğrulanması (Affirming Consequent) biçimsel safsatasını test eder."""
    onculler = [
        "Eğer yağmur yağarsa yerler ıslanır.",
        "Yerler şu an ıslak.",
    ]
    sonuc = "O halde kesinlikle yağmur yağdı."

    rapor = SafsataTespitcisi.safsata_tara(onculler, sonuc)
    assert rapor["safsata_var_mi"] is True
    assert rapor["safsata_anahtari"] == "affirming_consequent"


def test_tumdengelim_motoru_sokrates_gecerli_ve_saglam():
    """Klasik Sokrates kıyasının hem geçerli (Valid) hem sağlam (Sound) olduğunu test eder."""
    motor = TumdengelimMotoru()
    arguman = "Tüm insanlar ölümlüdür. Sokrates bir insandır. Dolayısıyla Sokrates ölümlüdür."
    
    degerlendirme = motor.argumani_degerlendir(arguman)
    assert degerlendirme["gecerli_mi"] is True
    assert degerlendirme["saglam_mi"] is True
    assert degerlendirme["safsata_bilgisi"]["safsata_var_mi"] is False
    assert degerlendirme["guven_skoru"] == 1.0


def test_gorsellestirici_pano_uretme():
    """6 panelli mantık ve safsata teşhis panosunun PNG olarak başarıyla kaydedildiğini test eder."""
    motor = TumdengelimMotoru()
    argumanlar = [
        "Tüm insanlar ölümlüdür. Sokrates bir insandır. Dolayısıyla Sokrates ölümlüdür.",
        "Sen zaten cahilsin. Dolayısıyla dediğin yanlıştır.",
    ]
    degerlendirmeler = [motor.argumani_degerlendir(a) for a in argumanlar]
    gorsellestirici = MantikGorsellestirici(dpi=100)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_mantik_pano.png")
        gorsellestirici.pano_olustur(degerlendirmeler, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
