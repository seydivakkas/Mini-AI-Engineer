"""
Day 91: Canlı AI Gözlemlenebilirlik ve İzleme Motoru (Observability Engine)
-------------------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, List, Optional, Tuple, Any
import time
import numpy as np
import torch

from .model import VisionModelObservability
from .metrik_toplayici import MetrikToplayici, MetrikOzeti
from .drift_dedektoru import DriftDedektoru, DriftRaporu


class AIObservabilityMotoru:
    """
    Üretim seviyesinde model çıkarımlarını gerçek zamanlı gözlemleyen,
    gecikme, işlem hacmi, veri kayması ve hata oranlarını tek çatı altında toplayan motor.
    """

    def __init__(
        self,
        model: VisionModelObservability,
        cihaz: str = "cpu",
        sla_gecikme_esigi_ms: float = 25.0,
        kayan_pencere_boyutu: int = 500,
    ):
        self.model = model.to(cihaz).eval()
        self.cihaz = cihaz
        self.kayan_pencere_boyutu = kayan_pencere_boyutu

        self.metrik_toplayici = MetrikToplayici(sla_gecikme_esigi_ms=sla_gecikme_esigi_ms)
        self.drift_dedektoru = DriftDedektoru()

        # Canlı tampon bellek (Sliding Window Buffer)
        self._canli_oznitelik_tamponu: List[np.ndarray] = []
        self._canli_tahmin_tamponu: List[int] = []
        self._canli_guven_tamponu: List[float] = []

    def referans_egitimi_yapilandir(self, referans_veriler: torch.Tensor) -> None:
        """Referans (baseline) veriyi modelden geçirerek referans dağılım profilini çıkarır."""
        self.model.eval()
        referans_veriler = referans_veriler.to(self.cihaz)

        with torch.no_grad():
            logitler, ozellikler = self.model(referans_veriler, ozellik_dondur=True)
            olasiliklar = torch.softmax(logitler, dim=-1)
            guvenler, siniflar = torch.max(olasiliklar, dim=-1)

        ref_ozellik_np = ozellikler.cpu().numpy()
        ref_tahmin_np = guvenler.cpu().numpy()

        self.drift_dedektoru.referans_belirle(
            referans_oznitelikler=ref_ozellik_np,
            referans_tahminler=ref_tahmin_np,
        )

    def tahmin_ve_gozlemle(self, girdi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Model çıkarımını icra eder, gecikmeyi hassas olarak ölçer ve gözlem metriklerini günceller.
        """
        if girdi.ndim == 3:
            girdi = girdi.unsqueeze(0)

        girdi = girdi.to(self.cihaz)
        t_baslangic = time.perf_counter()
        hata = False

        try:
            with torch.no_grad():
                logitler, ozellikler = self.model(girdi, ozellik_dondur=True)
                if self.cihaz == "cuda":
                    torch.cuda.synchronize()

                olasiliklar = torch.softmax(logitler, dim=-1)
                guvenler, siniflar = torch.max(olasiliklar, dim=-1)
        except Exception as e:
            hata = True
            raise e
        finally:
            t_bitis = time.perf_counter()
            gecikme_ms = (t_bitis - t_baslangic) * 1000.0
            self.metrik_toplayici.kayit_ekle(gecikme_ms=gecikme_ms, hata_olustu=hata)

        # Tampon belleğe ekleme
        ozellik_np = ozellikler.cpu().numpy()
        guven_np = guvenler.cpu().numpy()
        sinif_np = siniflar.cpu().numpy()

        for i in range(len(girdi)):
            self._canli_oznitelik_tamponu.append(ozellik_np[i])
            self._canli_guven_tamponu.append(float(guven_np[i]))
            self._canli_tahmin_tamponu.append(int(sinif_np[i]))

        # Tamponu sınırla
        if len(self._canli_oznitelik_tamponu) > self.kayan_pencere_boyutu:
            fazlalik = len(self._canli_oznitelik_tamponu) - self.kayan_pencere_boyutu
            self._canli_oznitelik_tamponu = self._canli_oznitelik_tamponu[fazlalik:]
            self._canli_guven_tamponu = self._canli_guven_tamponu[fazlalik:]
            self._canli_tahmin_tamponu = self._canli_tahmin_tamponu[fazlalik:]

        return siniflar.cpu(), guvenler.cpu(), gecikme_ms

    def kayma_raporu_al(self) -> Optional[DriftRaporu]:
        """Canlı tampon bellekteki veriler üzerinde drift analizini çalıştırır."""
        if len(self._canli_oznitelik_tamponu) < 10:
            return None

        canli_matris = np.array(self._canli_oznitelik_tamponu)
        canli_guven = np.array(self._canli_guven_tamponu)

        return self.drift_dedektoru.analiz_et(
            canli_oznitelikler=canli_matris,
            canli_tahminler=canli_guven,
        )

    def sistem_saglik_durumu_al(self) -> Dict[str, Any]:
        """Genel sistem sağlığı ve alarm durumunu döner."""
        metrik_ozeti: MetrikOzeti = self.metrik_toplayici.ozet_rapor_uret()
        drift_raporu: Optional[DriftRaporu] = self.kayma_raporu_al()

        alarmlar = []
        if metrik_ozeti.sla_ihlal_orani > 0.05:
            alarmlar.append(f"SLA İhlal Oranı Yüksek: %{metrik_ozeti.sla_ihlal_orani * 100:.1f}")
        if metrik_ozeti.hata_orani > 0.01:
            alarmlar.append(f"İstek Hata Oranı Kritik: %{metrik_ozeti.hata_orani * 100:.1f}")
        if drift_raporu and drift_raporu.genel_durum == "KRITIK_KAYMA":
            alarmlar.append(f"Kritik Veri Kayması Tespit Edildi! (Kayan Oran: %{drift_raporu.sistem_drift_orani * 100:.1f})")

        return {
            "metrikler": metrik_ozeti,
            "drift": drift_raporu,
            "alarmlar": alarmlar,
            "durum": "SAGLIKLI" if not alarmlar else "ALARM_VERILDI",
        }
