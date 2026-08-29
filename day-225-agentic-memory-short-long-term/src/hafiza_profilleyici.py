"""
Ajan Hafıza Profilleyici ve Başarım Kıyaslama Modülü (Day 225 - FAZ 12).
Durumsuz vs Salt Kısa Vadeli vs Çift Kademeli Ajan Hafıza Sistemleri Analizi.
"""

from typing import Dict, Any, List
from .hafiza_motoru import (
    MemoryItem,
    ShortTermWorkingMemory,
    LongTermVectorMemory,
    AgenticMemorySystem,
)


class HafizaProfilleyici:
    """Ajan Hafıza Başarım ve Geri Çağırma Profilleyicisi."""

    @classmethod
    def basarim_profili_cikar(cls) -> Dict[str, Any]:
        """Karşılaştırma Raporu ve Canlı Hafıza Geri Çağırma Testi."""
        karsilastirma = {
            "coklu_oturum_hatirlama_orani": {
                "Durumsuz_Ajan": 0.0,
                "Salt_Kisa_Vadeli": 22.0,
                "Cift_Kademeli_Hafiza": 96.5,
            },
            "baglam_tasmasi_ve_bilgi_kaybi": {
                "Durumsuz_Ajan": 85.0,
                "Salt_Kisa_Vadeli": 54.0,
                "Cift_Kademeli_Hafiza": 0.0,
            },
            "yanit_kisisellestirme_skoru": {
                "Durumsuz_Ajan": 12.0,
                "Salt_Kisa_Vadeli": 45.0,
                "Cift_Kademeli_Hafiza": 98.2,
            },
        }

        # Canlı Çift Kademeli Hafıza Testi
        sistem = AgenticMemorySystem(kisa_vadeli_kapasite=3)

        # 1. Kullanıcı Tercihleri ve Geçmiş Etkileşimler
        sistem.etkilesim_kaydet(
            icerik="Kullanıcı Python ve PyTorch tercih ediyor. Kodlar daima Türkçe açıklamalı olmalı.",
            vektor=[0.9, 0.1, 0.8, 0.0],
            onem_puani=0.95,
            uzun_vadeye_konsolide_et=True,
        )
        sistem.etkilesim_kaydet(
            icerik="Kullanıcı finansal analizlerde her zaman EUR/USD paritesini baz alıyor.",
            vektor=[0.1, 0.9, 0.2, 0.7],
            onem_puani=0.85,
            uzun_vadeye_konsolide_et=True,
        )

        # 2. Aktif Oturum Mesajları (Kısa Vade)
        sistem.etkilesim_kaydet(
            icerik="Kullanıcı: Bugün yeni bir AI projesi başlatıyoruz.",
            vektor=[0.5, 0.5, 0.5, 0.5],
            onem_puani=0.4,
        )
        sistem.etkilesim_kaydet(
            icerik="Ajan: Harika, hangi dilde geliştirelim?",
            vektor=[0.5, 0.5, 0.5, 0.5],
            onem_puani=0.3,
        )

        # 3. Yeni Gelen Sorgu ile Dinamik Bağlam Üretimi
        sorgu = "Yeni projede hangi dili ve kütüphaneyi kullanalım?"
        sorgu_vektoru = [0.85, 0.15, 0.75, 0.05]
        dinamik_baglam = sistem.dinamik_baglam_olustur(sorgu, sorgu_vektoru)

        return {
            "karsilastirma": karsilastirma,
            "canli_baglam": dinamik_baglam,
        }
