"""
Test-Time Compute Hesaplayıcı ve Bütçe Dağıtım Modülü (Day 147 - Faz 8).
Derinlik (Depth) vs Genişlik (Width) hesaplama ticareti ve kaynak tahsisi.
"""

from typing import Dict, Any, List
import numpy as np


class TestTimeHesaplayici:
    """Çıkarım zamanı hesaplama bütçesini derinlik ve genişlik boyutlarına paylaştıran modül."""

    @classmethod
    def butce_dagitimi_analiz_et(
        cls,
        toplam_token_butcesi: int,
        adim_basi_token: int = 128,
    ) -> Dict[str, Any]:
        """
        Sabit bir token bütçesinde 3 farklı stratejiyi analiz eder:
        1. Paralel Örnekleme (Geniş / Sığ)
        2. Derin Sıralı İnceleme (Dar / Derin)
        3. Dengeli MCTS Ağaç Araması (Optimal)
        """
        toplam_adim_kapasitesi = max(1, toplam_token_butcesi // adim_basi_token)

        # 1. Paralel Örnekleme (Self-Consistency): K yüksek, Derinlik = 1
        k_paralel = toplam_adim_kapasitesi
        dogruluk_paralel = float(np.clip(0.50 + 0.35 * (1.0 - np.exp(-k_paralel / 12.0)), 0.0, 0.88))

        # 2. Derin Sıralı Arama (Single Deep Chain): K = 1, Derinlik yüksek
        d_sirali = toplam_adim_kapasitesi
        dogruluk_sirali = float(np.clip(0.50 + 0.30 * (1.0 - np.exp(-d_sirali / 10.0)), 0.0, 0.82))

        # 3. Dengeli Ağaç Araması (MCTS / ToT): K ve D dengeli
        k_agac = int(np.sqrt(toplam_adim_kapasitesi))
        d_agac = int(np.sqrt(toplam_adim_kapasitesi))
        dogruluk_agac = float(np.clip(0.50 + 0.45 * (1.0 - np.exp(-toplam_adim_kapasitesi / 8.0)), 0.0, 0.96))

        return {
            "toplam_token_butcesi": toplam_token_butcesi,
            "toplam_adim_kapasitesi": toplam_adim_kapasitesi,
            "paralel_ornekleme": {
                "ornek_sayisi_k": k_paralel,
                "derinlik_d": 1,
                "tahmini_dogruluk": round(dogruluk_paralel, 4),
            },
            "derin_sirali_arama": {
                "ornek_sayisi_k": 1,
                "derinlik_d": d_sirali,
                "tahmini_dogruluk": round(dogruluk_sirali, 4),
            },
            "dengeli_agac_aramasi": {
                "genislik_k": max(1, k_agac),
                "derinlik_d": max(1, d_agac),
                "tahmini_dogruluk": round(dogruluk_agac, 4),
            },
        }
