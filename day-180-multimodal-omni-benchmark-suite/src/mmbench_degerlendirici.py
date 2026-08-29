"""
MMBench ve CircularEval Değerlendirici Modülü (Day 180 - FAZ 9 FİNALİ).
Liu et al. (2023) MMBench: Is Your Multi-modal Model an All-around Player?
CircularEval: Seçenek pozisyon önyargısını ve rastgele tahminleri sıfırlayan 4 turlu dairesel permütasyon testi.
"""

from typing import Dict, Any, List
import numpy as np


class MMBenchDegerlendirici:
    """MMBench ve CircularEval Çok Turlu Tutarlılık Değerlendiricisi."""

    KATEGORILER = [
        "fine_grained_perception",
        "spatial_relationship",
        "attribute_reasoning",
        "action_recognition",
        "logic_reasoning",
        "cross_instance_reasoning"
    ]

    def __init__(self):
        pass

    @classmethod
    def circular_eval_sorusu_degerlendir(
        cls,
        turlar: List[int],  # 4 turun her birinde seçilen seçenek indeksi (0, 1, 2, 3)
        dogru_indeksler: List[int],  # Her turdaki doğru seçeneğin pozisyon indeksi
    ) -> Dict[str, Any]:
        """
        Bir soru için 4 turlu CircularEval sonucunu doğrular.
        - Tam Başarı: Model 4 turun tamamında doğru seçeneği seçmiş mi?
        - Tutarlılık: Model şıklara takılmadan her turda aynı seçeneği mi takip etti?
        """
        assert len(turlar) == 4 and len(dogru_indeksler) == 4, "CircularEval tam 4 tur gerektirir."

        tur_dogruluklari = [t == d for t, d in zip(turlar, dogru_indeksler)]
        tum_turlar_dogru = all(tur_dogruluklari)

        # Tutarlılık kontrolü: modelin seçtiği semantik seçenekler her turda eşleşti mi?
        # Eğer tur_dogruluklari tamamen True ise veya model hep aynı semantik seçeneğe gitti ise tutarlıdır
        tutarlilik = (all(tur_dogruluklari) or all(not td for td in tur_dogruluklari))

        return {
            "tam_dogru": tum_turlar_dogru,
            "tek_tur_dogruluk_orani": sum(tur_dogruluklari) / 4.0,
            "tutarlilik": tutarlilik,
            "tur_detaylari": tur_dogruluklari,
        }

    def toplu_mmbench_degerlendir(
        self,
        kategori_verileri: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Tüm kategoriler için MMBench CircularEval metriklerini derler.
        """
        kategori_sonuclari = {}
        toplam_soru = 0
        toplam_circular_dogru = 0
        toplam_vanilla_dogru = 0
        toplam_tutarlilik = 0

        for kat in self.KATEGORILER:
            sorular = kategori_verileri.get(kat, [])
            N = len(sorular)
            if N == 0:
                continue

            circ_dogru = sum(1 for s in sorular if s["tam_dogru"])
            vanilla_dogru = sum(s["tek_tur_dogruluk_orani"] for s in sorular)
            tutarlilik_sayisi = sum(1 for s in sorular if s["tutarlilik"])

            circ_acc = (circ_dogru / N) * 100.0
            vanilla_acc = (vanilla_dogru / N) * 100.0
            consist_rate = (tutarlilik_sayisi / N) * 100.0

            kategori_sonuclari[kat] = {
                "soru_sayisi": N,
                "circular_acc": round(circ_acc, 2),
                "vanilla_acc": round(vanilla_acc, 2),
                "tutarlilik_orani": round(consist_rate, 2),
                "pozisyon_onyargi_kaybi": round(vanilla_acc - circ_acc, 2),
            }

            toplam_soru += N
            toplam_circular_dogru += circ_dogru
            toplam_vanilla_dogru += vanilla_dogru
            toplam_tutarlilik += tutarlilik_sayisi

        genel_circ_acc = (toplam_circular_dogru / max(toplam_soru, 1)) * 100.0
        genel_vanilla_acc = (toplam_vanilla_dogru / max(toplam_soru, 1)) * 100.0
        genel_tutarlilik = (toplam_tutarlilik / max(toplam_soru, 1)) * 100.0

        return {
            "genel_circular_acc": round(genel_circ_acc, 2),
            "genel_vanilla_acc": round(genel_vanilla_acc, 2),
            "genel_tutarlilik_orani": round(genel_tutarlilik, 2),
            "pozisyon_onyargi_farki": round(genel_vanilla_acc - genel_circ_acc, 2),
            "kategori_detaylari": kategori_sonuclari,
            "toplam_degerlendirilen_soru": toplam_soru,
        }

    @classmethod
    def ornek_model_mmbench_raporu(cls, model_adi: str = "Mini-Omni-v1") -> Dict[str, Any]:
        """Örnek model için CircularEval değerlendirme simülasyonu."""
        if "GPT-4o" in model_adi:
            base_acc = 0.86
        elif "Claude" in model_adi or "Gemini" in model_adi:
            base_acc = 0.83
        elif "LLaVA" in model_adi or "Qwen" in model_adi:
            base_acc = 0.77
        else:
            base_acc = 0.72

        np.random.seed(hash(model_adi + "_mmbench") % 2**32)
        kategori_verileri = {}

        for kat in cls.KATEGORILER:
            sorular = []
            for _ in range(25):
                # Doğru seçenek indeksleri (0..3)
                dogru_idx = [np.random.randint(0, 4) for _ in range(4)]
                # Model tahminleri
                turlar = []
                for d in dogru_idx:
                    if np.random.rand() < base_acc:
                        turlar.append(d)
                    else:
                        turlar.append(np.random.randint(0, 4))

                eval_res = cls.circular_eval_sorusu_degerlendir(turlar, dogru_idx)
                sorular.append(eval_res)
            kategori_verileri[kat] = sorular

        evaluator = cls()
        return evaluator.toplu_mmbench_degerlendir(kategori_verileri)
