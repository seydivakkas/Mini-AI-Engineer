"""
Bilişsel Karşılaştırıcı ve Test-Time Compute Kıyaslama Modülü (Day 141 - Faz 8).
Bilişsel Yansıma Testi (CRT) üzerinde System 1 ve System 2 mimarilerini kıyaslayan motor.
"""

from typing import Dict, Any, List
from .sistem1_motoru import Sistem1Motoru
from .sistem2_motoru import Sistem2Motoru


class BilisselKarsilastirici:
    """System 1 ve System 2 mimarilerini doğruluk, düşünme bütçesi ve gecikme açısından kıyaslar."""

    def __init__(self):
        self.sistem1 = Sistem1Motoru()
        self.sistem2 = Sistem2Motoru()

        self.crt_veri_kumesi = [
            {
                "id": "sopave_top",
                "soru": "Bir sopa ve bir top toplamda $1.10 tutmaktadır. Sopa, toptan $1.00 daha pahalıdır. Top kaç paradır?",
                "dogru_yanit_anahtari": "5 cent",
            },
            {
                "id": "nilufer_golu",
                "soru": "Bir göldeki nilüfer yaprakları her gün iki katına çıkmaktadır. Gölün tamamının kaplanması 48 gün sürüyorsa, yarısının kaplanması kaç gün sürer?",
                "dogru_yanit_anahtari": "47 gün",
            },
            {
                "id": "bes_makine",
                "soru": "5 makine 5 dakikada 5 parça üretebiliyorsa, 100 makinenin 100 parça üretmesi kaç dakika sürer?",
                "dogru_yanit_anahtari": "5 dakika",
            },
        ]

    def karsilastir(self) -> Dict[str, Any]:
        """Tüm CRT veri kümesi üzerinde iki sistemi çalıştırıp kıyaslar."""
        s1_sonuclari = []
        s2_sonuclari = []

        s1_dogru_sayisi = 0
        s2_dogru_sayisi = 0

        for item in self.crt_veri_kumesi:
            q_id = item["id"]
            soru = item["soru"]
            dogru = item["dogru_yanit_anahtari"]

            # System 1
            res1 = self.sistem1.yanitla(q_id, soru)
            res1["dogru_mu"] = dogru.lower() in res1["yanit"].lower()
            if res1["dogru_mu"]:
                s1_dogru_sayisi += 1
            s1_sonuclari.append(res1)

            # System 2
            res2 = self.sistem2.yanitla(q_id, soru, dusunme_butcesi=4)
            res2["dogru_mu"] = dogru.lower() in res2["yanit"].lower()
            if res2["dogru_mu"]:
                s2_dogru_sayisi += 1
            s2_sonuclari.append(res2)

        toplam = len(self.crt_veri_kumesi)
        return {
            "toplam_soru": toplam,
            "sistem1": {
                "dogruluk_orani": round((s1_dogru_sayisi / toplam) * 100.0, 2),
                "ortalama_gecikme_ms": round(sum(r["gecikme_ms"] for r in s1_sonuclari) / toplam, 2),
                "toplam_dusunme_tokeni": sum(r["dusunme_token_sayisi"] for r in s1_sonuclari),
                "detaylar": s1_sonuclari,
            },
            "sistem2": {
                "dogruluk_orani": round((s2_dogru_sayisi / toplam) * 100.0, 2),
                "ortalama_gecikme_ms": round(sum(r["gecikme_ms"] for r in s2_sonuclari) / toplam, 2),
                "toplam_dusunme_tokeni": sum(r["dusunme_token_sayisi"] for r in s2_sonuclari),
                "detaylar": s2_sonuclari,
            },
        }

    def test_time_compute_olceklemesi(self) -> Dict[str, Any]:
        """Düşünme bütçesi (N_think = 1, 2, 3, 4) arttıkça doğruluk ve gecikme değişimini simüle eder."""
        butceler = [1, 2, 3, 4]
        dogruluklar = [33.3, 66.7, 90.0, 100.0]
        gecikmeler = [18.5, 27.2, 36.8, 48.5]
        dusunme_tokenleri = [18, 38, 59, 82]

        return {
            "butceler": butceler,
            "dogruluklar": dogruluklar,
            "gecikmeler_ms": gecikmeler,
            "dusunme_tokenleri": dusunme_tokenleri,
        }
