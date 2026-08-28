"""Model Hata Denetimi ve Başarısızlık Kök Neden Analizi Modülü.

Bu modül; modelin aşırı güvenle yanlış tahmin ettiği (Overconfident Failures),
kararsız kaldığı (Ambiguous / High-Entropy Cases) örnekleri ve en çok karıştırdığı
sınıf çiftlerini (Confusion Pairs) tespit ederek hata denetimi (Error Audit) yapar.
"""

from typing import Dict, List, Tuple
import numpy as np


class HataDenetcisi:
    """Modelin hatalı ve sınırda kalan tahminlerini denetleyen sınıf."""

    @staticmethod
    def asiri_guvenli_yanlislar(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        en_fazla: int = 5,
    ) -> List[Dict]:
        """Modelin yüksek olasılıkla yanlış tahmin ettiği örnekleri güvene göre azalan sırada döner."""
        y_pred = np.argmax(y_probs, axis=1)
        confidences = np.max(y_probs, axis=1)
        yanlis_mask = y_pred != y_true

        yanlis_indeksler = np.where(yanlis_mask)[0]
        if len(yanlis_indeksler) == 0:
            return []

        # Güvene göre azalan sırala
        sirali_indeksler = yanlis_indeksler[np.argsort(confidences[yanlis_indeksler])[::-1]]

        sonuclar = []
        for idx in sirali_indeksler[:en_fazla]:
            sonuclar.append({
                "ornek_indeks": int(idx),
                "gercek_sinif": int(y_true[idx]),
                "tahmin_sinif": int(y_pred[idx]),
                "guven": float(confidences[idx]),
                "tum_olasiliklar": y_probs[idx].tolist(),
            })
        return sonuclar

    @staticmethod
    def belirsiz_tahminler(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        en_fazla: int = 5,
    ) -> List[Dict]:
        """Modelin iki veya daha fazla sınıf arasında kaldığı (en düşük güvene sahip) örnekleri döner."""
        y_pred = np.argmax(y_probs, axis=1)
        confidences = np.max(y_probs, axis=1)

        sirali_indeksler = np.argsort(confidences)  # En düşük güvenden başla

        sonuclar = []
        for idx in sirali_indeksler[:en_fazla]:
            sonuclar.append({
                "ornek_indeks": int(idx),
                "gercek_sinif": int(y_true[idx]),
                "tahmin_sinif": int(y_pred[idx]),
                "guven": float(confidences[idx]),
                "dogru_mu": bool(y_pred[idx] == y_true[idx]),
                "tum_olasiliklar": y_probs[idx].tolist(),
            })
        return sonuclar

    @staticmethod
    def en_cok_karisan_ciftler(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sinif_isimleri: List[str],
    ) -> List[Tuple[str, str, int]]:
        """En sık birbirine karıştırılan (Gerçek A, Tahmin B) sınıf çiftlerini döner."""
        ciftler: Dict[Tuple[int, int], int] = {}

        for gercek, tahmin in zip(y_true, y_pred):
            if gercek != tahmin:
                key = (int(gercek), int(tahmin))
                ciftler[key] = ciftler.get(key, 0) + 1

        sirali_ciftler = sorted(ciftler.items(), key=lambda item: item[1], reverse=True)

        rapor = [
            (sinif_isimleri[g], sinif_isimleri[t], sayi)
            for (g, t), sayi in sirali_ciftler
        ]
        return rapor
