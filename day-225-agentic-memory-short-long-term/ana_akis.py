"""
Day 225: Ajan Hafıza Sistemleri (Kısa & Uzun Vadeli Bellek) Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.hafiza_motoru import (
    MemoryItem,
    ShortTermWorkingMemory,
    LongTermVectorMemory,
    AgenticMemorySystem,
)
from src.hafiza_profilleyici import HafizaProfilleyici
from src.gorsellestirici import HafizaGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 225 (FAZ 12): AJAN HAFIZA SİSTEMLERİ - KISA VADELİ ÇALIŞMA VE VEKTÖREL UZUN VADELİ EPİZODİK BELLEK")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Çift Kademeli Ajan Hafıza Sistemini Başlatma
    # -------------------------------------------------------------
    print("\n[1/4] Çift Kademeli Ajan Hafıza Sistemi Başlatılıyor...")
    hafiza_sistemi = AgenticMemorySystem(kisa_vadeli_kapasite=3)
    print("  ✓ Kısa Vadeli Çalışma Belleği (Kapasite: 3) & Vektörel Uzun Vadeli Epizodik Depo Hazırlandı.")

    # -------------------------------------------------------------
    # ADIM 2: Kullanıcı Tercihleri ve Epizodik Anıları Kaydetme
    # -------------------------------------------------------------
    print("\n[2/4] Kalıcı Kullanıcı Tercihleri ve Epizodik Bilgiler Konsolide Ediliyor...")

    # Hatıra 1: Python/PyTorch Tercihi
    hafiza_sistemi.etkilesim_kaydet(
        icerik="Kullanıcı: Tüm modeller PyTorch ile yazılmalı ve kod blokları detaylı Türkçe açıklamalar içermelidir.",
        vektor=[0.92, 0.08, 0.85, 0.02],
        onem_puani=0.95,
        uzun_vadeye_konsolide_et=True,
    )

    # Hatıra 2: Veritabanı ve Lisans Kuralı
    hafiza_sistemi.etkilesim_kaydet(
        icerik="Kullanıcı: Projelerde asla MIT lisansı kullanma; her zaman 'ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR' kuralına uy.",
        vektor=[0.15, 0.90, 0.05, 0.88],
        onem_puani=0.99,
        uzun_vadeye_konsolide_et=True,
    )
    print("  ✓ Uzun Vadeli Epizodik Anılar Başarıyla İndekslendi.")

    # -------------------------------------------------------------
    # ADIM 3: Yeni Sorgu ile Dinamik Hatırlama ve Bağlam Üretimi
    # -------------------------------------------------------------
    print("\n[3/4] Yeni Oturum Sorgusu İçin Anlamsal Benzerlik ve Yenilik Puanlı Geri Çağırma...")
    sorgu = "Yeni modülü geliştirirken hangi derin öğrenme kütüphanesini ve açıklama dilini kullanalım?"
    sorgu_vektoru = [0.88, 0.12, 0.80, 0.05]

    dinamik_baglam = hafiza_sistemi.dinamik_baglam_olustur(sorgu, sorgu_vektoru)
    print(dinamik_baglam)

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Ajan Hafıza Teşhis Panosu Oluşturuluyor...")
    profil_raporu = HafizaProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "ajan_hafiza_paneli.png")

    HafizaGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Ajan Hafıza Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 225 (FAZ 12): AJAN HAFIZA SİSTEMLERİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
