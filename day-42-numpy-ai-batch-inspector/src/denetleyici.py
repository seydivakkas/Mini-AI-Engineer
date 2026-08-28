"""
Yapay Zeka Batch Girdi Tensörü Denetim Motoru (NumPy AI Batch Inspector).
"""

from typing import Dict, Any, List
import time
import numpy as np
from .sema import TensorSemasi


class AIBatchDenetleyici:
    """Üretim modellerine giren NumPy tensörlerini donanım ve matematiksel kurallara göre teftiş eder."""

    def __init__(self, sema: TensorSemasi):
        self.sema = sema

    def denetle(self, tensor: Any) -> Dict[str, Any]:
        """Girdi tensörünü kapsamlı kontrolden geçirip telemetri ve hata raporu üretir."""
        baslangic_zaman = time.perf_counter()
        ihlaller = []
        duzeltilebilir = True

        # 1. Temel Dizi ve Tip Kontrolü
        if not isinstance(tensor, np.ndarray):
            try:
                tensor = np.asarray(tensor)
                ihlaller.append({
                    "kod": "TIP_UYARISI",
                    "mesaj": "Girdi doğrudan numpy.ndarray değil; otomatik dönüştürüldü.",
                    "kritik": False
                })
            except Exception as e:
                return {
                    "karar": "KRITIK_RED",
                    "gecerli": False,
                    "hata": f"Tensöre dönüştürülemedi: {str(e)}",
                    "ihlaller": [{"kod": "FATAL_TYPE_ERROR", "mesaj": str(e), "kritik": True}]
                }

        dtype = tensor.dtype
        if dtype not in self.sema.gecerli_tipler:
            ihlaller.append({
                "kod": "GECERSIZ_DTYPE",
                "mesaj": f"Geçersiz veri tipi: {dtype}. Beklenen: {[str(t) for t in self.sema.gecerli_tipler]}",
                "kritik": False
            })

        # Bellek Boyutu ve C-Contiguity
        bellek_mb = float(tensor.nbytes / (1024.0 * 1024.0))
        if bellek_mb > self.sema.max_bellek_mb:
            ihlaller.append({
                "kod": "BELLEK_ASIMI",
                "mesaj": f"Tensör bellek limiti aşıldı! {bellek_mb:.2f} MB > {self.sema.max_bellek_mb} MB",
                "kritik": True
            })
            duzeltilebilir = False

        c_contiguous = bool(tensor.flags['C_CONTIGUOUS'])
        if self.sema.sureklilik_sarti and not c_contiguous:
            ihlaller.append({
                "kod": "BELLEK_DUZENI_UYARISI",
                "mesaj": "Tensör bellekte C-contiguous (kesintisiz) sırada değil.",
                "kritik": False
            })

        # 2. Şekil ve Kanal Düzeni Kontrolü
        sekil_uygun, sekil_mesaj = self.sema.sekil_kurali.dogrula(tensor.shape)
        nhwc_tespit_edildi = False

        if not sekil_uygun:
            # NHWC -> NCHW terslik kontrolü (Örn: (B, 224, 224, 3) geldi mi?)
            if len(tensor.shape) == 4 and tensor.shape[-1] == 3 and self.sema.sekil_kurali.kanal_sirasi == "NCHW":
                nhwc_tespit_edildi = True
                ihlaller.append({
                    "kod": "KANAL_DUZENI_TERS",
                    "mesaj": f"Kanal düzeni NHWC olarak tespit edildi {tensor.shape}; model NCHW bekliyor.",
                    "kritik": False
                })
            else:
                ihlaller.append({
                    "kod": "SEKIL_UYUSMAZLIGI",
                    "mesaj": sekil_mesaj,
                    "kritik": True
                })
                duzeltilebilir = False

        # 3. Sayısal Kararsızlık (NaN & Inf) Kontrolü
        nan_sayisi = int(np.isnan(tensor).sum())
        inf_sayisi = int(np.isinf(tensor).sum())

        if nan_sayisi > 0:
            ihlaller.append({
                "kod": "NAN_DEGERI_TESPITI",
                "mesaj": f"Tensörde {nan_sayisi} adet NaN (Tanımsız Sayı) tespit edildi!",
                "kritik": True
            })
            duzeltilebilir = False

        if inf_sayisi > 0:
            ihlaller.append({
                "kod": "INF_DEGERI_TESPITI",
                "mesaj": f"Tensörde {inf_sayisi} adet ±Sonsuz (Inf) taşma tespit edildi!",
                "kritik": True
            })
            duzeltilebilir = False

        # 4. Değer Aralığı ve İstatistiksel Dağılım
        min_v = float(np.min(tensor)) if nan_sayisi == 0 and inf_sayisi == 0 else float("nan")
        max_v = float(np.max(tensor)) if nan_sayisi == 0 and inf_sayisi == 0 else float("nan")
        mean_v = float(np.mean(tensor)) if nan_sayisi == 0 and inf_sayisi == 0 else float("nan")
        std_v = float(np.std(tensor)) if nan_sayisi == 0 and inf_sayisi == 0 else float("nan")

        alt_sinir, ust_sinir = self.sema.deger_araligi
        aralik_disi_sayisi = 0
        if nan_sayisi == 0 and inf_sayisi == 0:
            aralik_disi_sayisi = int(np.sum((tensor < alt_sinir) | (tensor > ust_sinir)))
            if aralik_disi_sayisi > 0:
                ihlaller.append({
                    "kod": "ARALIK_DISI_DEGERLER",
                    "mesaj": f"{aralik_disi_sayisi} adet değer [{alt_sinir}, {ust_sinir}] aralığı dışında! (Min: {min_v:.2f}, Max: {max_v:.2f})",
                    "kritik": False
                })

            # Doygunluk / Sabit Tensör Kontrolü (Kamera körleşmesi / kapalı sensör)
            if std_v < 1e-6:
                ihlaller.append({
                    "kod": "SABIT_TENSOR_UYARISI",
                    "mesaj": f"Tensör varyansı 0'a yakın (Std: {std_v:.6f}); sensör arızası veya boş görüntü olabilir.",
                    "kritik": False
                })

        # 5. Karar Mekanizması
        gecen_sure_ms = float((time.perf_counter() - baslangic_zaman) * 1000.0)
        kritik_sayisi = sum(1 for ih in ihlaller if ih["kritik"])

        if len(ihlaller) == 0:
            karar = "GECERLI"
            guvenli_gecis = True
        elif kritik_sayisi > 0 or not duzeltilebilir:
            karar = "KRITIK_RED"
            guvenli_gecis = False
        else:
            karar = "DUZELTILEBILIR_UYARI"
            guvenli_gecis = True

        return {
            "karar": karar,
            "guvenli_gecis": guvenli_gecis,
            "denetim_suresi_ms": float(round(gecen_sure_ms, 3)),
            "sekil": list(tensor.shape),
            "dtype": str(dtype),
            "bellek_mb": float(round(bellek_mb, 3)),
            "c_contiguous": c_contiguous,
            "nhwc_tespit_edildi": nhwc_tespit_edildi,
            "istatistikler": {
                "min": float(round(min_v, 3)) if not np.isnan(min_v) else None,
                "max": float(round(max_v, 3)) if not np.isnan(max_v) else None,
                "ortalama": float(round(mean_v, 3)) if not np.isnan(mean_v) else None,
                "standart_sapma": float(round(std_v, 3)) if not np.isnan(std_v) else None,
                "nan_sayisi": nan_sayisi,
                "inf_sayisi": inf_sayisi,
                "aralik_disi_piksel": aralik_disi_sayisi
            },
            "toplam_ihlal_sayisi": len(ihlaller),
            "ihlaller": ihlaller
        }
