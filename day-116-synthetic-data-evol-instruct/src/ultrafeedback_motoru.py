"""
UltraFeedback Çok Boyutlu Tercih Değerlendirme Modülü (Day 116).
4 boyutlu (Talimat Takibi, Doğruluk, Faydalılık, Derinlik) puanlama ve DPO/SimPO/ORPO çiftli tercih veri üretimi.
"""

from typing import Dict, List, Tuple, Any
import random


class UltraFeedbackPuanlayici:
    """Aday yanıtları 4 eksende puanlayıp (x, y_w, y_l) tercih çiftleri üreten motor."""

    BOYUTLAR = [
        "Talimat Takibi (Instruction Following)",
        "Teknik Doğruluk (Factuality & Correctness)",
        "Faydalılık & Açıklık (Helpfulness)",
        "Muhakeme Derinliği (Reasoning Depth)",
    ]

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def yaniti_puanla(self, prompt: str, yanit: str) -> Dict[str, Any]:
        """Bir yanıtı 4 boyutta (1-5 puan) analiz eder ve toplam bileşik skoru hesaplar."""
        # Basit kural tabanlı değerlendirme simülatörü
        k_kisit = 1 if "[KISIT]" in prompt else 0
        uzunluk = len(yanit.split())

        # 1. Talimat Takibi (1-5)
        p_talimat = 5.0 if ("def " in yanit or "```" in yanit or uzunluk > 20) else 3.0
        # 2. Teknik Doğruluk (1-5)
        p_dogruluk = 4.5 if "O(1)" in yanit or "karmaşıklık" in yanit else 3.5
        # 3. Faydalılık (1-5)
        p_fayda = min(5.0, 2.5 + (uzunluk / 15.0))
        # 4. Muhakeme Derinliği (1-5)
        p_derinlik = 5.0 if ("Adım 1" in yanit or "Örnek" in yanit) else 3.0

        bilesik_skor = (p_talimat * 0.35 + p_dogruluk * 0.25 + p_fayda * 0.20 + p_derinlik * 0.20)

        return {
            "talimat_takibi": p_talimat,
            "dogruluk": p_dogruluk,
            "faydalilik": p_fayda,
            "muhakeme_derinligi": p_derinlik,
            "toplam_skor": bilesik_skor,
        }

    def tercih_cifti_uret(
        self,
        prompt: str,
        aday_yanitlar: List[str],
    ) -> Dict[str, Any]:
        """Aday yanıtlar arasından en iyiyi (chosen) ve en zayıfı (rejected) seçip çift üretir."""
        puanli_adaylar = []
        for yanit in aday_yanitlar:
            puanlar = self.yaniti_puanla(prompt, yanit)
            puanli_adaylar.append((yanit, puanlar))

        # Puana göre sırala (Büyükten küçüğe)
        puanli_adaylar.sort(key=lambda x: x[1]["toplam_skor"], reverse=True)

        chosen_yanit, chosen_metrik = puanli_adaylar[0]
        rejected_yanit, rejected_metrik = puanli_adaylar[-1]

        elestiri = (
            f"Chosen Skor: {chosen_metrik['toplam_skor']:.2f} (Talimat: {chosen_metrik['talimat_takibi']}, Derinlik: {chosen_metrik['muhakeme_derinligi']}) | "
            f"Rejected Skor: {rejected_metrik['toplam_skor']:.2f} (Yetersiz teknik detay ve kısıt ihmali)."
        )

        return {
            "prompt": prompt,
            "chosen": chosen_yanit,
            "rejected": rejected_yanit,
            "chosen_skor": chosen_metrik["toplam_skor"],
            "rejected_skor": rejected_metrik["toplam_skor"],
            "elestiri": elestiri,
            "detaylar": {
                "chosen_metrikler": chosen_metrik,
                "rejected_metrikler": rejected_metrik,
            },
        }
