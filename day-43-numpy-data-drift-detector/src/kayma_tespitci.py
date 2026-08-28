"""
Çok Boyutlu Veri Kayması ve MLOps Alarm Yöneticisi (Data Drift Detector).
"""

from typing import Dict, Any, List, Optional
import numpy as np
from .dagilim_olcer import KSVeWassersteinHesaplayici


class VeriKaymasiDedektoru:
    """Üretim modellerinin girdi dağılımlarını izleyerek veri kayması alarmları üretir."""

    def __init__(
        self,
        referans_veriler: Dict[str, np.ndarray],
        alpha: float = 0.05,
        kritik_psi_esigi: float = 0.20
    ):
        self.referans_veriler = {k: np.asarray(v, dtype=np.float64) for k, v in referans_veriler.items()}
        self.alpha = alpha
        self.kritik_psi_esigi = kritik_psi_esigi

    def teftis_et(
        self,
        uretim_verileri: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Tüm öznitelikler üzerinde veri kayması analizini yürütür."""
        oznitelik_raporlari = {}
        kayan_oznitelik_sayisi = 0
        kritik_kayma_sayisi = 0

        for oznitelik_adi, ref_dizi in self.referans_veriler.items():
            if oznitelik_adi not in uretim_verileri:
                continue

            prod_dizi = np.asarray(uretim_verileri[oznitelik_adi], dtype=np.float64)
            analiz = KSVeWassersteinHesaplayici.olc(ref_dizi, prod_dizi, alpha=self.alpha)

            oznitelik_raporlari[oznitelik_adi] = analiz

            if analiz["drift_tespit_edildi"]:
                kayan_oznitelik_sayisi += 1
            if analiz["kayma_derecesi"] == "KRITIK_KAYMA_ALARM":
                kritik_kayma_sayisi += 1

        toplam_oznitelik = len(oznitelik_raporlari)
        kayma_orani = float(kayan_oznitelik_sayisi / max(toplam_oznitelik, 1))

        # MLOps Karar Mekanizması
        if kritik_kayma_sayisi > 0 or kayma_orani >= 0.50:
            genel_durum = "KRITIK_VERI_KAYMASI_ALARM"
            mlops_aksiyonu = "YENIDEN_EGITIM_BORU_HATTINI_TETIKLE (TRIGGER_RETRAIN)"
            alarm = True
        elif kayan_oznitelik_sayisi > 0:
            genel_durum = "ORTA_DUZEY_KAYMA_UYARISI"
            mlops_aksiyonu = "SIKILASTIRILMIS_IZLEME_MODU (MONITOR_CLOSELY)"
            alarm = False
        else:
            genel_durum = "DAGILIMLAR_KARARLI_NORMAL"
            mlops_aksiyonu = "EYLEM_GEREKMIYOR_MODEL_SAGLIKLI (NO_ACTION)"
            alarm = False

        return {
            "genel_durum": genel_durum,
            "mlops_aksiyonu": mlops_aksiyonu,
            "alarm_verildi": alarm,
            "toplam_oznitelik_sayisi": toplam_oznitelik,
            "kayan_oznitelik_sayisi": kayan_oznitelik_sayisi,
            "kritik_kayma_sayisi": kritik_kayma_sayisi,
            "kayma_orani": float(round(kayma_orani * 100.0, 2)),
            "oznitelikler": oznitelik_raporlari
        }
