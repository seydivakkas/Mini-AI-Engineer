"""
Taslak Yanıt ve Olgusal İddia Üretici Modülü (Day 156 - Faz 8).
Kullanıcı sorgusuna ilk taslak yanıtı (Baseline Response) üretir ve içerdiği olgusal iddiaları çıkarır.
"""

from typing import Dict, Any, List


class TaslakUreticisi:
    """İlk taslak yanıtı ve iddia listesini oluşturan ajan."""

    @classmethod
    def taslak_uret(cls, soru: str) -> Dict[str, Any]:
        """
        Soruya ham bir başlangıç taslağı üretir ve doğrulanacak iddiaları listeler.
        """
        # Tipik bir halüsinasyon içeren ilk taslak simülasyonu
        taslak_metin = (
            "Mehmet Akif Ersoy 1873 yılında Ankara'da doğmuştur. "
            "İstiklal Marşı'nı Çankaya Köşkü'nde kaleme almış ve "
            "marş 1923 yılında TBMM tarafından milli marş olarak kabul edilmiştir."
        )

        iddialar = [
            {"iddia_id": "c1", "konu": "Doğum Yeri", "taslak_ifade": "Ankara'da doğmuştur", "iddia_turu": "Yer"},
            {"iddia_id": "c2", "konu": "Yazıldığı Mekan", "taslak_ifade": "Çankaya Köşkü'nde kaleme almıştır", "iddia_turu": "Mekan"},
            {"iddia_id": "c3", "konu": "Kabul Yılı", "taslak_ifade": "1923 yılında kabul edilmiştir", "iddia_turu": "Tarih"},
        ]

        return {
            "kullanici_sorusu": soru,
            "taslak_yanit": taslak_metin,
            "iddialar": iddialar,
            "iddia_sayisi": len(iddialar),
        }
