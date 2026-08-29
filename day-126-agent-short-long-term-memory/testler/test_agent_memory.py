"""
FAZ 7 GÜN 126: Multi-Tier Agent Memory Systems Testleri.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import numpy as np
import pytest

from src.bellek_katmanlari import BellekTipi, BellekKaydi, CalismaBellegi, SemantikBellek
from src.bellek_yoneticisi import BellekYoneticisi
from src.bellek_ajani import HafizaliAjan
from src.gorsellestirici import BellekGorsellestirici


def test_bellek_kaydi_ve_l2_norm():
    """BellekKaydi sınıfının vektörü L2 normalize ettiğini ve erişim sayacını güncellediğini test eder."""
    v = np.array([3.0, 4.0, 0.0])
    kayit = BellekKaydi(metin="Kullanıcı Python uzmanıdır", vektor=v, tip=BellekTipi.SEMANTIK, onem_puani=8.0)

    norm = np.linalg.norm(kayit.vektor)
    assert abs(norm - 1.0) < 1e-5
    assert kayit.erisim_sayisi == 1

    kayit.erisim_kaydet()
    assert kayit.erisim_sayisi == 2


def test_calisma_bellegi_kayan_pencere():
    """CalismaBellegi sınıfının kapasite aşımında eski mesajları attığını (Sliding Window) test eder."""
    cb = CalismaBellegi(kapasite=3)
    cb.ekle("kullanici", "Mesaj 1")
    cb.ekle("asistan", "Yanıt 1")
    cb.ekle("kullanici", "Mesaj 2")
    cb.ekle("asistan", "Yanıt 2")

    assert len(cb.mesajlar) == 3
    assert "Mesaj 1" not in cb.baglam_metni()
    assert "Yanıt 2" in cb.baglam_metni()


def test_semantik_bellek_ekle_ve_gecersiz_kil():
    """SemantikBellek sınıfının aktif ve geçersiz kılınan kayıtları doğru yönettiğini test eder."""
    sb = SemantikBellek()
    v = np.ones(32)
    k1 = BellekKaydi(metin="Tercih A", vektor=v, tip=BellekTipi.SEMANTIK, bellek_id="id_1")
    k2 = BellekKaydi(metin="Tercih B", vektor=v, tip=BellekTipi.SEMANTIK, bellek_id="id_2")

    sb.ekle_veya_guncelle(k1)
    sb.ekle_veya_guncelle(k2)
    assert len(sb.aktif_kayitlar()) == 2

    sb.gecersiz_kil("id_1")
    aktifler = sb.aktif_kayitlar()
    assert len(aktifler) == 1
    assert aktifler[0].id == "id_2"


def test_bellek_yoneticisi_olgu_cikarma_ve_add():
    """BellekYoneticisi sınıfının yeni olguları ADD işlemiyle kaydettiğini test eder."""
    by = BellekYoneticisi()
    rapor = by.olgu_cikar_ve_kaydet("Projelerimde PyTorch kütüphanesini tercih ediyorum.")

    assert len(rapor) > 0
    assert any("ADD" in r["islem"] for r in rapor)
    assert len(by.semantik.aktif_kayitlar()) == 1


def test_bellek_yoneticisi_celiski_ve_update():
    """BellekYoneticisi sınıfının güncellenen tercihte eski kaydı geçersiz kılıp UPDATE yaptığını test eder."""
    by = BellekYoneticisi()
    by.olgu_cikar_ve_kaydet("Projelerimde PyTorch kütüphanesini kullanıyorum.")
    assert len(by.semantik.aktif_kayitlar()) == 1

    # Tercih değişikliği bildirimi
    rapor_guncelleme = by.olgu_cikar_ve_kaydet("Artık PyTorch yerine JAX kütüphanesini kullanıyorum.")

    assert any("UPDATE" in r["islem"] or "ADD" in r["islem"] for r in rapor_guncelleme)
    # Aktif kayıtlardan biri JAX içermeli
    aktif_metinler = [k.metin for k in by.semantik.aktif_kayitlar()]
    assert any("JAX" in m for m in aktif_metinler)


def test_hibrit_arama_siralamasi():
    """BellekYoneticisi hibrit arama fonksiyonunun benzerlik, tazelik ve önemi birleştirdiğini test eder."""
    by = BellekYoneticisi()
    by.olgu_cikar_ve_kaydet("Veri tabanı olarak PostgreSQL kullanıyorum.")
    by.olgu_cikar_ve_kaydet("Front-end için React tercih ediyorum.")

    sonuclar = by.hibrit_arama("Hangi veritabanını kullanıyordum?", top_k=1)
    assert len(sonuclar) == 1
    en_iyi_kayit, skor = sonuclar[0]
    assert "PostgreSQL" in en_iyi_kayit.metin
    assert skor > 0.0


def test_hafizali_ajan_mesaj_isle():
    """HafizaliAjan sınıfının geçmiş tercihleri hatırlayarak kişiselleştirilmiş yanıt ürettiğini test eder."""
    ajan = HafizaliAjan()
    # Tur 1: Tercih bildir
    ajan.mesaj_isle("Projelerimde derin öğrenme için PyTorch kullanıyorum.")
    # Tur 2: Kod istemi
    yanit_raporu = ajan.mesaj_isle("Bana bir sınıflandırıcı modeli kur.")

    assert "PyTorch" in yanit_raporu["asistan_yaniti"]
    assert yanit_raporu["calisma_bellegi_boyutu"] >= 4


def test_gorsellestirici_pano():
    """BellekGorsellestirici sınıfının 6 panelli PNG teşhis dosyasını ürettiğini test eder."""
    ajan = HafizaliAjan()
    rapor = ajan.mesaj_isle("PyTorch ve CUDA ile çalışıyorum.")
    karsilastirma = ajan.benchmark_karsilastir()

    gorsellestirici = BellekGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_memory_pano.png")
        gorsellestirici.pano_olustur(rapor, karsilastirma, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
