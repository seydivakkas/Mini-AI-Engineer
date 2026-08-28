"""
Eşik Değeri Mühendisliği, F-Beta Optimizasyonu ve Maliyet-Fayda Karar Motoru (Threshold Engineering Engine).
"""

from typing import Dict, Any, List, Tuple
import numpy as np


class EsikDegeriMuhendisi:
    """F-Beta, Finansal Net Kazanç ve Karar Eğrisi Analizi (DCA) ile optimal olasılık eşiklerini hesaplar."""

    @classmethod
    def esik_tarama_analizi(
        cls,
        y_gercek: np.ndarray,
        y_olasilik: np.ndarray,
        maliyet_matrisi: Dict[str, float] = None,
        esik_sayisi: int = 100
    ) -> Dict[str, Any]:
        """Tüm eşikler üzerinde F-Beta, Net Kazanç, Karmaşıklık Matrisi hücreleri ve DCA hesaplar."""
        y_true = np.asarray(y_gercek, dtype=int)
        y_prob = np.asarray(y_olasilik, dtype=float)
        N = len(y_true)

        if maliyet_matrisi is None:
            maliyet_matrisi = {
                "b_tp": 2500.0,   # Yakalanan risk faydası ($)
                "b_tn": 25.0,     # Sorunsuz işlem faydası ($)
                "c_fp": 150.0,    # Yanlış alarm / manuel inceleme maliyeti ($)
                "c_fn": 4000.0    # Kaçırılan risk / dolandırıcılık zararı ($)
            }

        esikler = np.linspace(0.01, 0.99, esik_sayisi)

        tp_list, tn_list, fp_list, fn_list = [], [], [], []
        f05_list, f1_list, f2_list = [], [], []
        net_kazanc_list = []
        dca_net_benefit = []

        b_tp = maliyet_matrisi.get("b_tp", 2500.0)
        b_tn = maliyet_matrisi.get("b_tn", 25.0)
        c_fp = maliyet_matrisi.get("c_fp", 150.0)
        c_fn = maliyet_matrisi.get("c_fn", 4000.0)

        for th in esikler:
            pred = (y_prob >= th).astype(int)
            tp = int(np.sum((y_true == 1) & (pred == 1)))
            tn = int(np.sum((y_true == 0) & (pred == 0)))
            fp = int(np.sum((y_true == 0) & (pred == 1)))
            fn = int(np.sum((y_true == 1) & (pred == 0)))

            tp_list.append(tp)
            tn_list.append(tn)
            fp_list.append(fp)
            fn_list.append(fn)

            p = tp / max(tp + fp, 1e-8)
            r = tp / max(tp + fn, 1e-8)

            # F-Beta Hesapları
            # F0.5
            f05 = (1 + 0.25) * (p * r) / max(0.25 * p + r, 1e-8)
            # F1
            f1 = 2.0 * (p * r) / max(p + r, 1e-8)
            # F2
            f2 = (1 + 4.0) * (p * r) / max(4.0 * p + r, 1e-8)

            f05_list.append(float(f05))
            f1_list.append(float(f1))
            f2_list.append(float(f2))

            # Finansal Net Kazanç / Fayda
            net_kazanc = (tp * b_tp) + (tn * b_tn) - (fp * c_fp) - (fn * c_fn)
            net_kazanc_list.append(float(net_kazanc))

            # Decision Curve Analysis (DCA) Net Benefit
            nb = (tp / N) - (fp / N) * (th / max(1.0 - th, 1e-6))
            dca_net_benefit.append(float(nb))

        # Optimal Noktaların Tespiti
        opt_f05_idx = int(np.argmax(f05_list))
        opt_f1_idx = int(np.argmax(f1_list))
        opt_f2_idx = int(np.argmax(f2_list))
        opt_kazanc_idx = int(np.argmax(net_kazanc_list))

        return {
            "esikler": esikler,
            "tp": np.array(tp_list),
            "tn": np.array(tn_list),
            "fp": np.array(fp_list),
            "fn": np.array(fn_list),
            "f05_skorlari": np.array(f05_list),
            "f1_skorlari": np.array(f1_list),
            "f2_skorlari": np.array(f2_list),
            "net_kazanc_listesi": np.array(net_kazanc_list),
            "dca_net_benefit": np.array(dca_net_benefit),
            "optimal_f05_esigi": float(round(esikler[opt_f05_idx], 3)),
            "optimal_f1_esigi": float(round(esikler[opt_f1_idx], 3)),
            "optimal_f2_esigi": float(round(esikler[opt_f2_idx], 3)),
            "optimal_finansal_esik": float(round(esikler[opt_kazanc_idx], 3)),
            "maksimum_net_kazanc": float(round(net_kazanc_list[opt_kazanc_idx], 2)),
            "maliyet_matrisi": maliyet_matrisi
        }
