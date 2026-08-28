"""
Sektörel RAG ve Reçete Danışman Motoru (ISO Standartları & Otomatik Hata Çözümü).
"""

from typing import List, Dict, Any, Optional
import re
import numpy as np


class SektorelRAGMotoru:
    """Teknik tekstil standartları üzerinde hibrit RAG araması ve hata çözüm önericisi."""

    BILGI_TABANI = [
        {
            "id": "DOC-YARN-01",
            "baslik": "TS EN ISO 2060 İplik Büküm ve Mukavemet Standardı",
            "standart": "TS EN ISO 2060",
            "anahtar_kelimeler": ["iplik", "büküm", "mukavemet", "kopma", "twist", "alpha_m", "dtex"],
            "cozum": "İplik kopmasını önlemek için büküm katsayısı Alpha_m = 75-85 aralığında tutulmalı, asgari mukavemet 14.5 cN/tex olmalıdır."
        },
        {
            "id": "DOC-WEAVE-02",
            "baslik": "Jakarlı Dokuma Çözgü Gerginlik ve Ayar Kılavuzu",
            "standart": "Vandewiele Dokuma Reçetesi",
            "anahtar_kelimeler": ["çözgü", "gerginlik", "tansiyon", "kopuk", "atkı", "jakar", "iplik_kopmasi"],
            "cozum": "Zemin çözgü tansiyonu 45 ± 5 cN aralığına çekilmeli, 55 cN üzerindeki aşırı gerilimler derhal düşürülmelidir."
        },
        {
            "id": "DOC-MAINT-03",
            "baslik": "Dokuma Tezgahı Yağ Lekesi ve Temizleme Talimatı",
            "standart": "Mekanik Bakım El Kitabı",
            "anahtar_kelimeler": ["yağ", "leke", "boya", "temizleme", "gres", "ultrasonik", "yag_boya_lekesi"],
            "cozum": "Damlayan taze yağ lekesine anında ultrasonik tabanca ile trikloretilensiz ekolojik leke çözücü uygulanmalı ve PTFE renksiz gres kullanılmalıdır."
        },
        {
            "id": "DOC-FINISH-04",
            "baslik": "Kurutma Fiksaj ve Apre Kimyasalları Standartları",
            "standart": "Tekstil Kimyası Terbiye Kılavuzu",
            "anahtar_kelimeler": ["apre", "sıcaklık", "fiksaj", "kurutma", "stenter", "florokarbon", "haslık"],
            "cozum": "Kurutma tüneli sıcaklığı 145°C - 155°C arasında sabitlenmeli, C6-florokarbon konsantrasyonu %2.5 - 3.0 uygulanmalıdır."
        }
    ]

    def hata_icin_cozum_getir(self, kusur_turu: str) -> Dict[str, Any]:
        """Tespit edilen kusur türüne göre standart fabrika çözümünü getirir."""
        tur_kucuk = kusur_turu.lower()
        for dok in self.BILGI_TABANI:
            for kw in dok["anahtar_kelimeler"]:
                if kw in tur_kucuk:
                    return {
                        "standart_id": dok["id"],
                        "standart_adi": dok["standart"],
                        "baslik": dok["baslik"],
                        "oneri": dok["cozum"]
                    }
        return {
            "standart_id": "GENEL-STANDART",
            "standart_adi": "ISO 9001 Kalite Kontrol",
            "baslik": "Genel Dokuma Kalite İncelemesi",
            "oneri": "Numuneyi kalite kontrol laboratuvarına sevk edip mikroskop altında iplik yapısını inceleyiniz."
        }

    def soru_sor(self, soru: str) -> Dict[str, Any]:
        """Soruya en uygun standardı ve alıntıyı getirir."""
        tokens = set(re.findall(r"\w+", soru.lower()))
        en_iyi_dok = None
        en_yuksek_skor = 0

        for dok in self.BILGI_TABANI:
            dok_tokens = set(dok["anahtar_kelimeler"] + re.findall(r"\w+", dok["baslik"].lower()))
            kesisim = len(tokens.intersection(dok_tokens))
            if kesisim > en_yuksek_skor:
                en_yuksek_skor = kesisim
                en_iyi_dok = dok

        if en_yuksek_skor == 0 or en_iyi_dok is None:
            return {
                "durum": "REDDEDILDI",
                "yanit": "Tekstil teknik bilgi tabanında bu parametreye dair doğrulanmış standart bulunmamaktadır."
            }

        return {
            "durum": "BASARILI",
            "standart": en_iyi_dok["standart"],
            "baslik": en_iyi_dok["baslik"],
            "yanit": en_iyi_dok["cozum"]
        }
