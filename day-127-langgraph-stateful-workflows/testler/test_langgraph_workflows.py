"""
FAZ 7 GÜN 127: LangGraph Stateful Workflows & Human-in-the-Loop Testleri.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.cizge_durumu import DurumIndirgeyici, varsayilan_durum_olustur
from src.kontrol_noktasi_yoneticisi import CheckpointYoneticisi
from src.cizge_motoru import DurumsalCizge, END
from src.is_akislari import iade_akisi_olustur
from src.gorsellestirici import LangGraphGorsellestirici


def test_durum_indirgeyici_liste_ve_skaler():
    """DurumIndirgeyici sınıfının mesaj listelerini ekleyip skaler değerleri güncellediğini test eder."""
    eski = {"mesajlar": [{"rol": "kullanici", "icerik": "Merhaba"}], "risk": 0.1}
    guncelleme = {"mesajlar": [{"rol": "asistan", "icerik": "Nasıl yardımcı olabilirim?"}], "risk": 0.5}

    yeni = DurumIndirgeyici.indirge(eski, guncelleme)
    assert len(yeni["mesajlar"]) == 2
    assert yeni["risk"] == 0.5


def test_varsayilan_durum_olustur():
    """varsayilan_durum_olustur fonksiyonunun tüm durum alanlarını doğru başlattığını test eder."""
    durum = varsayilan_durum_olustur(musteri_id="M_99", talep_tutari=1200.0, baslangic_mesaji="İade istiyorum")

    assert durum["musteri_id"] == "M_99"
    assert durum["talep_tutari"] == 1200.0
    assert len(durum["mesajlar"]) == 1
    assert durum["adim_gecmisi"] == ["START"]


def test_checkpoint_yoneticisi_kaydet_ve_geri_sar():
    """CheckpointYoneticisi sınıfının durum kopyalarını kaydedip başarıyla geri sarabildiğini (Rollback) test eder."""
    cy = CheckpointYoneticisi()
    d1 = {"adim": 1, "deger": 100}
    d2 = {"adim": 2, "deger": 200}

    cy.kaydet(1, "DugumA", d1)
    cy.kaydet(2, "DugumB", d2)

    assert len(cy.gecmis) == 2
    geri_sarilan = cy.geri_sar(1)
    assert geri_sarilan["deger"] == 100


def test_durumsal_cizge_duz_akis():
    """DurumsalCizge sınıfının koşulsuz düz kenarları doğru sırayla çalıştırdığını test eder."""
    cizge = DurumsalCizge()

    def n1(d):
        return {"sayac": d.get("sayac", 0) + 10}

    def n2(d):
        return {"sayac": d.get("sayac", 0) + 5}

    cizge.add_node("N1", n1)
    cizge.add_node("N2", n2)
    cizge.set_entry_point("N1")
    cizge.add_edge("N1", "N2")
    cizge.add_edge("N2", END)

    sonuc = cizge.calistir({"sayac": 0})
    assert sonuc["tamamlandi"] is True
    assert sonuc["durum"]["sayac"] == 15


def test_durumsal_cizge_kosullu_yonlendirme():
    """DurumsalCizge sınıfının duruma göre doğru dallara koşullu yönlendiğini test eder."""
    cizge = DurumsalCizge()

    def n_basla(d):
        return {"puan": d.get("girdi_puan", 0)}

    def n_pozitif(d):
        return {"sonuc_metni": "POZITIF"}

    def n_negatif(d):
        return {"sonuc_metni": "NEGATIF"}

    def router(d):
        return "pos" if d.get("puan", 0) >= 0 else "neg"

    cizge.add_node("Basla", n_basla)
    cizge.add_node("PozitifDugum", n_pozitif)
    cizge.add_node("NegatifDugum", n_negatif)
    cizge.set_entry_point("Basla")

    cizge.add_conditional_edges("Basla", router, {"pos": "PozitifDugum", "neg": "NegatifDugum"})
    cizge.add_edge("PozitifDugum", END)
    cizge.add_edge("NegatifDugum", END)

    res_pos = cizge.calistir({"girdi_puan": 10})
    assert res_pos["durum"]["sonuc_metni"] == "POZITIF"

    res_neg = cizge.calistir({"girdi_puan": -5})
    assert res_neg["durum"]["sonuc_metni"] == "NEGATIF"


def test_iade_akisi_dusuk_risk_otomatik_onay():
    """İade iş akışının düşük riskli tutarlarda insan onayı beklemeden otomatik ödendiğini test eder."""
    akisi = iade_akisi_olustur()
    baslangic = varsayilan_durum_olustur(musteri_id="M_1", talep_tutari=1500.0)

    sonuc = akisi.calistir(baslangic)
    assert sonuc["tamamlandi"] is True
    assert sonuc["durum"]["odeme_yapildi_mi"] is True
    assert "OdemeIadesi" in sonuc["durum"]["adim_gecmisi"]
    assert "InsanOnayi" not in sonuc["durum"]["adim_gecmisi"]


def test_iade_akisi_yuksek_risk_hitl_kesinti_ve_devam():
    """İade iş akışının yüksek riskli tutarlarda HITL kesintisine uğrayıp onay sonrası ödendiğini test eder."""
    akisi = iade_akisi_olustur()
    baslangic = varsayilan_durum_olustur(musteri_id="M_VIP", talep_tutari=8500.0)

    # 1. Faz: Kesinti Bekle
    kesinti_sonucu = akisi.calistir(baslangic)
    assert kesinti_sonucu["tamamlandi"] is False
    assert kesinti_sonucu["kesinti_noktasi"] == "InsanOnayi"
    assert kesinti_sonucu["durum"]["nihai_durum"] == "BEKLIYOR_INSAN_ONAYI"

    # 2. Faz: İnsan Denetçi Onayı ile Devam Et
    durdurulan_durum = kesinti_sonucu["durum"]
    nihai_sonuc = akisi.calistir(durdurulan_durum, insan_onay_yaniti=True)

    assert nihai_sonuc["tamamlandi"] is True
    assert nihai_sonuc["durum"]["odeme_yapildi_mi"] is True
    assert nihai_sonuc["durum"]["insan_onayladi_mi"] is True


def test_gorsellestirici_pano():
    """LangGraphGorsellestirici sınıfının 6 panelli PNG teşhis dosyasını ürettiğini test eder."""
    akisi = iade_akisi_olustur()
    baslangic = varsayilan_durum_olustur(musteri_id="M_PANO", talep_tutari=2500.0)
    sonuc = akisi.calistir(baslangic)
    gecmis = akisi.checkpoint_yoneticisi.gecmis_ozeti()

    gorsellestirici = LangGraphGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_langgraph_pano.png")
        gorsellestirici.pano_olustur(sonuc, gecmis, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
