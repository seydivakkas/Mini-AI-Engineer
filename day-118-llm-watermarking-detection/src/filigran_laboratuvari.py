"""
LLM Filigranlama ve Tespit Kıyaslama Laboratuvarı Modülü (Day 118).
Filigranlı vs Filigransız vs İnsan metinleri, delta duyarlılığı ve Paraphrase dayanıklılık testleri.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import random

from .filigran_ekleyici import KirchenbauerWatermarker
from .filigran_tespitci import WatermarkDetector


class FiligranLaboratuvari:
    """Filigranlama ve tespit performansını istatistiksel olarak doğrulayan deney laboratuvarı."""

    def __init__(
        self,
        vocab_size: int = 1000,
        gamma: float = 0.5,
        delta: float = 2.5,
        gizli_anahtar: int = 15485863,
        z_esigi: float = 4.0,
        seed: int = 42,
    ):
        random.seed(seed)
        self.ekleyici = KirchenbauerWatermarker(vocab_size=vocab_size, gamma=gamma, delta=delta, gizli_anahtar=gizli_anahtar)
        self.tespitci = WatermarkDetector(vocab_size=vocab_size, gamma=gamma, gizli_anahtar=gizli_anahtar, z_esigi=z_esigi)
        self.vocab_size = vocab_size

    def benchmark_kostur(self, ornek_sayisi: int = 30, dizi_uzunlugu: int = 100) -> Dict[str, Any]:
        """Filigranlı ve Filigransız metin grupları için Z-Skoru ve tespit doğruluğunu hesaplar."""
        filigranli_z = []
        filigransiz_z = []
        filigranli_yesil_oran = []
        filigransiz_yesil_oran = []

        filigranli_dogru_tespit = 0
        filigransiz_yanlis_alarm = 0

        for i in range(ornek_sayisi):
            # 1. Filigranlı Metin Üretimi (Delta = 2.5)
            f_dizi = self.ekleyici.token_dizisi_uret(baslangic_token=random.randint(0, self.vocab_size - 1), uzunluk=dizi_uzunlugu, filigran_aktif=True)
            f_analiz = self.tespitci.filigran_analizi(f_dizi)
            filigranli_z.append(f_analiz["z_skoru"])
            filigranli_yesil_oran.append(f_analiz["yesil_oran"])
            if f_analiz["filigran_var_mi"]:
                filigranli_dogru_tespit += 1

            # 2. Filigransız Metin Üretimi (Delta = 0)
            nf_dizi = self.ekleyici.token_dizisi_uret(baslangic_token=random.randint(0, self.vocab_size - 1), uzunluk=dizi_uzunlugu, filigran_aktif=False)
            nf_analiz = self.tespitci.filigran_analizi(nf_dizi)
            filigransiz_z.append(nf_analiz["z_skoru"])
            filigransiz_yesil_oran.append(nf_analiz["yesil_oran"])
            if nf_analiz["filigran_var_mi"]:
                filigransiz_yanlis_alarm += 1

        # 3. Delta Duyarlılık Analizi
        delta_degerleri = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        delta_z_skorlari = []
        for d in delta_degerleri:
            gecici_ekleyici = KirchenbauerWatermarker(vocab_size=self.vocab_size, gamma=self.ekleyici.gamma, delta=d, gizli_anahtar=self.ekleyici.gizli_anahtar)
            z_toplam = 0.0
            for _ in range(10):
                d_dizi = gecici_ekleyici.token_dizisi_uret(baslangic_token=10, uzunluk=dizi_uzunlugu, filigran_aktif=True)
                z_toplam += self.tespitci.filigran_analizi(d_dizi)["z_skoru"]
            delta_z_skorlari.append(z_toplam / 10.0)

        # 4. Paraphrase Saldırı Dayanıklılığı (Rastgele Token Değiştirme)
        edit_oranlari = [0.0, 0.15, 0.30, 0.45, 0.60]
        paraphrase_z = []
        for edit_oran in edit_oranlari:
            z_ort = 0.0
            for _ in range(10):
                dizi = self.ekleyici.token_dizisi_uret(baslangic_token=20, uzunluk=dizi_uzunlugu, filigran_aktif=True)
                # Rastgele token değiştirme
                degisen = list(dizi)
                for idx in range(1, len(degisen)):
                    if random.random() < edit_oran:
                        degisen[idx] = random.randint(0, self.vocab_size - 1)
                z_ort += self.tespitci.filigran_analizi(degisen)["z_skoru"]
            paraphrase_z.append(z_ort / 10.0)

        return {
            "filigranli_z": filigranli_z,
            "filigransiz_z": filigransiz_z,
            "filigranli_ort_z": float(np.mean(filigranli_z)),
            "filigransiz_ort_z": float(np.mean(filigransiz_z)),
            "filigranli_yesil_oran": float(np.mean(filigranli_yesil_oran)),
            "filigransiz_yesil_oran": float(np.mean(filigransiz_yesil_oran)),
            "tpr_dogru_tespit_orani": (filigranli_dogru_tespit / ornek_sayisi) * 100.0,
            "fpr_yanlis_alarm_orani": (filigransiz_yanlis_alarm / ornek_sayisi) * 100.0,
            "delta_degerleri": delta_degerleri,
            "delta_z_skorlari": delta_z_skorlari,
            "edit_oranlari": edit_oranlari,
            "paraphrase_z": paraphrase_z,
        }
