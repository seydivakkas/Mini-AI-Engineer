"""
Omni Benchmark Liderlik Tablosu ve Çok Modlu Model Karşılaştırıcı Modülü (Day 180 - FAZ 9 FİNALİ).
MME, MMBench, MathVista ve POPE (Nesne Halüsinasyon Dayanıklılığı) metriklerini birleştiren merkezi değerlendirme motoru.
"""

from typing import Dict, Any, List
from .mme_degerlendirici import MMEDegerlendirici
from .mmbench_degerlendirici import MMBenchDegerlendirici
from .mathvista_degerlendirici import MathVistaDegerlendirici


class OmniBenchmarkMerkezi:
    """Çok Modlu Modeller İçin Bütünleşik Liderlik Tablosu ve Skorlama Merkezi."""

    BENCHMARK_AGIRLIKLARI = {
        "mme_norm": 0.30,       # MME Algı & Biliş (2800 üzerinden %'ye normalize)
        "mmbench": 0.30,        # MMBench CircularEval Sağlamlık Doğruluğu
        "mathvista": 0.25,      # MathVista Görsel Matematik Akıl Yürütme
        "pope_f1": 0.15,        # POPE Nesne Halüsinasyon F1 Skoru
    }

    MODEL_LISTESI = [
        "GPT-4o (OpenAI 2024)",
        "Claude 3.5 Sonnet (Anthropic 2024)",
        "Gemini 1.5 Pro (Google 2024)",
        "Qwen2-VL-72B (Alibaba 2024)",
        "LLaVA-NeXT-34B (Liu et al. 2024)",
        "Mini-Omni-v1 (FAZ 9 Capstone 2026)",
    ]

    def __init__(self):
        pass

    @classmethod
    def pope_halusinasyon_skoru_uret(cls, model_adi: str) -> Dict[str, float]:
        """POPE (Polling-based Object Probing Evaluation) Nesne Halüsinasyon Metrikleri."""
        if "GPT-4o" in model_adi:
            f1, acc, yes_ratio = 91.2, 90.8, 51.2
        elif "Claude" in model_adi or "Gemini" in model_adi:
            f1, acc, yes_ratio = 89.5, 89.1, 52.0
        elif "Qwen" in model_adi:
            f1, acc, yes_ratio = 88.0, 87.5, 52.8
        elif "LLaVA" in model_adi:
            f1, acc, yes_ratio = 86.4, 85.9, 54.1
        else:  # Mini-Omni-v1
            f1, acc, yes_ratio = 84.8, 84.2, 55.0

        return {
            "f1_skoru": f1,
            "dogruluk": acc,
            "yes_orani": yes_ratio,
            "halusinasyon_orani": round(100.0 - acc, 2),
        }

    def tek_model_omni_skor_hesapla(self, model_adi: str) -> Dict[str, Any]:
        """Tek bir model için 4 benchmark'ı çalıştırır ve ağırlıklı Omni-Score üretir."""
        mme_res = MMEDegerlendirici.ornek_model_mme_raporu(model_adi)
        mmbench_res = MMBenchDegerlendirici.ornek_model_mmbench_raporu(model_adi)
        mathvista_res = MathVistaDegerlendirici.ornek_model_mathvista_raporu(model_adi)
        pope_res = self.pope_halusinasyon_skoru_uret(model_adi)

        mme_norm = mme_res["genel_basari_yuzdesi"]
        mmbench_score = mmbench_res["genel_circular_acc"]
        mathvista_score = mathvista_res["genel_mathvista_skoru"]
        pope_score = pope_res["f1_skoru"]

        omni_score = (
            self.BENCHMARK_AGIRLIKLARI["mme_norm"] * mme_norm +
            self.BENCHMARK_AGIRLIKLARI["mmbench"] * mmbench_score +
            self.BENCHMARK_AGIRLIKLARI["mathvista"] * mathvista_score +
            self.BENCHMARK_AGIRLIKLARI["pope_f1"] * pope_score
        )

        return {
            "model_adi": model_adi,
            "omni_score": round(omni_score, 2),
            "mme": {
                "toplam_puan": mme_res["toplam_mme_skoru"],
                "perception": mme_res["perception_skoru"],
                "cognition": mme_res["cognition_skoru"],
                "norm_yuzde": mme_norm,
            },
            "mmbench": {
                "circular_acc": mmbench_score,
                "vanilla_acc": mmbench_res["genel_vanilla_acc"],
                "tutarlilik": mmbench_res["genel_tutarlilik_orani"],
            },
            "mathvista": {
                "dogruluk": mathvista_score,
            },
            "pope": pope_res,
        }

    def tum_modelleri_karsilastir(self) -> Dict[str, Any]:
        """Tüm öncü modeller için tam liderlik tablosu oluşturur."""
        sonuclar = []
        for model in self.MODEL_LISTESI:
            res = self.tek_model_omni_skor_hesapla(model)
            sonuclar.append(res)

        # Omni skoruna göre sırala (Büyükten küçüğe)
        sonuclar.sort(key=lambda x: x["omni_score"], reverse=True)

        for rank, item in enumerate(sonuclar, 1):
            item["siralama"] = rank

        return {
            "liderlik_tablosu": sonuclar,
            "toplam_model_sayisi": len(sonuclar),
            "en_yuksek_skorlu_model": sonuclar[0]["model_adi"],
            "faz_9_capstone_modeli": [m for m in sonuclar if "Mini-Omni" in m["model_adi"]][0],
        }
