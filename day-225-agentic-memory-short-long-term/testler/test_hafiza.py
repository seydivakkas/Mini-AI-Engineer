"""
PyTest Birim Testleri - Day 225: Ajan Hafıza Sistemleri Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.hafiza_motoru import (
    MemoryItem,
    ShortTermWorkingMemory,
    LongTermVectorMemory,
    AgenticMemorySystem,
)
from src.hafiza_profilleyici import HafizaProfilleyici
from src.gorsellestirici import HafizaGorsellestirici


def test_memory_item_creation():
    """1. MemoryItem nesnesi alanları doğru tip ve tensörle başlatmalıdır."""
    item = MemoryItem(1, "Önemli not", [1.0, 0.0, 0.0], 0.9, 5, "EPIZODIK")
    assert item.hafiza_id == 1
    assert item.onem_puani == 0.9
    assert item.vektor.shape == (3,)


def test_short_term_memory_fifo():
    """2. ShortTermWorkingMemory kapasite aşıldığında en eski elemanı FIFO ile çıkarmalıdır."""
    kisa = ShortTermWorkingMemory(kapasite=2)
    kisa.ekle("Mesaj 1")
    kisa.ekle("Mesaj 2")
    tasan = kisa.ekle("Mesaj 3")
    assert tasan == "Mesaj 1"
    assert len(kisa.tampon) == 2
    assert "Mesaj 3" in kisa.baglam_metni()


def test_long_term_memory_cosine_similarity():
    """3. LongTermVectorMemory iki vektörün kosinüs benzerliğini doğru hesaplamalıdır."""
    uzun = LongTermVectorMemory()
    v1 = np.array([1.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.0, 1.0])
    assert pytest.approx(uzun.benzerlik_hesapla(v1, v2), 0.001) == 1.0
    assert pytest.approx(uzun.benzerlik_hesapla(v1, v3), 0.001) == 0.0


def test_weighted_retrieval_ranking():
    """4. Uzun vadeli bellek sorgusu birleşik skora göre doğru sıralama yapmalıdır."""
    uzun = LongTermVectorMemory()
    uzun.ekle("Alakasız Hatıra", [0.0, 1.0], onem=0.1, zaman=1)
    uzun.ekle("Tam Eşleşen Önemli Hatıra", [1.0, 0.0], onem=0.9, zaman=10)

    sonuclar = uzun.sorgula([1.0, 0.0], mevcut_zaman=10, top_k=1)
    assert len(sonuclar) == 1
    assert sonuclar[0][0].icerik == "Tam Eşleşen Önemli Hatıra"
    assert sonuclar[0][1] > 0.8


def test_agentic_memory_consolidation():
    """5. AgenticMemorySystem önemli etkileşimleri uzun vadeli depoya konsolide etmelidir."""
    sistem = AgenticMemorySystem(kisa_vadeli_kapasite=2)
    sistem.etkilesim_kaydet("Kritik Bilgi", [0.5, 0.5], onem_puani=0.9, uzun_vadeye_konsolide_et=True)
    assert len(sistem.uzun_bellek.hafiza_havuzu) == 1


def test_dynamic_context_assembly():
    """6. Dinamik bağlam oluşturucu hem kısa hem uzun vadeli bilgileri içermelidir."""
    sistem = AgenticMemorySystem()
    sistem.etkilesim_kaydet("Kullanıcı Adı: Seydi", [1.0, 0.0], onem_puani=0.9, uzun_vadeye_konsolide_et=True)
    baglam = sistem.dinamik_baglam_olustur("Kullanıcı kimdir?", [1.0, 0.0])
    assert "Seydi" in baglam
    assert "AJAN HAFIZA RAPORU" in baglam


def test_profiler_memory_metrics():
    """7. Profilleyici çift kademeli hafızanın çoklu oturum başarısının %90 üstünde olduğunu göstermelidir."""
    prof = HafizaProfilleyici.basarim_profili_cikar()
    skor = prof["karsilastirma"]["coklu_oturum_hatirlama_orani"]["Cift_Kademeli_Hafiza"]
    assert skor > 90.0


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. HafizaGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_hafiza_paneli.png")
    profil = HafizaProfilleyici.basarim_profili_cikar()

    HafizaGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
