"""
Bağımsız Doğrulama ve Fakt Kontrol Modülü (Day 156 - Faz 8).
Soruları ilk taslaktan bağımsız olarak cevaplar ve olgusal doğruları belirler.
"""

from typing import Dict, Any, List


class BagimsizDogrulayici:
    """Doğrulama sorularını önyargısız izole bağlamda yanıtlayan motor."""

    GERCEK_VERITABANI = {
        "Doğum Yeri": {
            "dogrulanmis_cevap": "İstanbul (Fatih, Sarıgüzel mahallesi)",
            "kanit": "Tarihi nüfus kayıtları ve biyografileri",
            "dogru_deger": "İstanbul",
        },
        "Yazıldığı Mekan": {
            "dogrulanmis_cevap": "Taceddin Dergâhı (Ankara / Altındağ)",
            "kanit": "Dergah kayıtları ve Mehmet Akif müzesi",
            "dogru_deger": "Taceddin Dergâhı",
        },
        "Kabul Yılı": {
            "dogrulanmis_cevap": "12 Mart 1921 (TBMM oturumu)",
            "kanit": "TBMM Zabıt Ceridesi 1921",
            "dogru_deger": "12 Mart 1921",
        },
    }

    @classmethod
    def sorulari_yanitla(cls, sorular: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Soruları yanıtlayarak taslak ile kıyaslanacak gerçekleri döner.
        """
        yanitlar = []

        for s in sorular:
            konu = s["konu"]
            gercek = cls.GERCEK_VERITABANI.get(
                konu,
                {"dogrulanmis_cevap": "Bilinmiyor", "kanit": "Genel Kaynak", "dogru_deger": "Yok"}
            )

            # Taslaktaki iddia gerçekle örtüşüyor mu?
            taslak = s["taslak_iddia"].lower()
            dogru_val = gercek["dogru_deger"].lower()
            celiski_var_mi = (dogru_val not in taslak)

            yanitlar.append({
                "iddia_id": s["iddia_id"],
                "konu": konu,
                "soru": s["dogrulama_sorusu"],
                "taslak_iddia": s["taslak_iddia"],
                "dogrulanmis_cevap": gercek["dogrulanmis_cevap"],
                "kanit": gercek["kanit"],
                "celiski_var_mi": celiski_var_mi,
                "durum": "DÜZELTİLDİ (Halüsinasyon)" if celiski_var_mi else "ONAYLANDI (Doğru)",
            })

        return yanitlar
