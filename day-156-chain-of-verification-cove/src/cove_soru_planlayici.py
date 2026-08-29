"""
CoVe Doğrulama Sorusu Planlayıcı Modülü (Day 156 - Faz 8).
Taslaktaki iddiaları teyit etmek için tarafsız, bağımsız çapraz kontrol soruları planlar.
"""

from typing import Dict, Any, List


class CoVESoruPlanlayici:
    """Olgusal iddialardan tarafsız doğrulama soruları türeten modül."""

    @classmethod
    def sorulari_planla(cls, iddialar: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Her iddia için taslaktaki cevabı içermeyen bağımsız bir soru üretir.
        """
        dogrulama_sorulari = []

        soru_eslemeleri = {
            "Doğum Yeri": "Mehmet Akif Ersoy hangi şehirde ve semtte doğmuştur?",
            "Yazıldığı Mekan": "Mehmet Akif Ersoy İstiklal Marşı'nı Ankara'da hangi tarihi mekanda/dergahta yazmıştır?",
            "Kabul Yılı": "İstiklal Marşı TBMM tarafından tam olarak hangi gün, ay ve yılda kabul edilmiştir?",
        }

        for iddia in iddialar:
            konu = iddia["konu"]
            soru_metni = soru_eslemeleri.get(konu, f"{konu} hakkındaki kesin tarihi gerçek nedir?")

            dogrulama_sorulari.append({
                "iddia_id": iddia["iddia_id"],
                "konu": konu,
                "dogrulama_sorusu": soru_metni,
                "taslak_iddia": iddia["taslak_ifade"],
            })

        return dogrulama_sorulari
