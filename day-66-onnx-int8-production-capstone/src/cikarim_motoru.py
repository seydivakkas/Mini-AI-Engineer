"""
ONNX Runtime Cikarim Motoru (Inference Engine)
==============================================
Optimize edilmis ONNX ve INT8 modellerini yuksek basarimli,
coklu is parcacikli (multi-threaded) calistiran cikarim yoneticisi.
"""

from typing import Dict, List, Optional, Union, Tuple
import os
import time
import numpy as np
import onnxruntime as ort


class ONNXInferenceEngine:
    """
    ONNX Runtime Inference Session yoneticisi.
    """

    def __init__(
        self,
        model_yolu: str,
        is_parcacigi_sayisi: int = 4,
        saglayici: Optional[str] = None
    ) -> None:
        if not os.path.exists(model_yolu):
            raise FileNotFoundError(f"Model dosyasi bulunamadi: {model_yolu}")

        self.model_yolu = model_yolu
        self.is_parcacigi_sayisi = is_parcacigi_sayisi

        # Oturum Ayarlari
        self.oturum_ayarlari = ort.SessionOptions()
        self.oturum_ayarlari.intra_op_num_threads = is_parcacigi_sayisi
        self.oturum_ayarlari.inter_op_num_threads = 1
        self.oturum_ayarlari.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.oturum_ayarlari.enable_cpu_mem_arena = True

        # Saglayicilar
        kullanilabilir_saglayicilar = ort.get_available_providers()
        if saglayici and saglayici in kullanilabilir_saglayicilar:
            secilen_saglayicilar = [saglayici]
        else:
            secilen_saglayicilar = ["CPUExecutionProvider"]

        self.oturum = ort.InferenceSession(
            self.model_yolu,
            sess_options=self.oturum_ayarlari,
            providers=secilen_saglayicilar
        )

        self.girdi_adi = self.oturum.get_inputs()[0].name
        self.girdi_sekli = self.oturum.get_inputs()[0].shape
        self.cikti_adi = self.oturum.get_outputs()[0].name
        self.cikti_sekli = self.oturum.get_outputs()[0].shape

    def isinma_yap(self, ornek_sekil: Tuple[int, int, int, int] = (1, 3, 64, 64), tekrar: int = 5) -> None:
        """
        Grafik optimizasyonlari ve onbellek ilklendirmesi icin sahte tensörle isinma kosar.
        """
        sahte_girdi = np.random.randn(*ornek_sekil).astype(np.float32)
        for _ in range(tekrar):
            self.tahmin_et(sahte_girdi)

    def tahmin_et(self, girdi_dizisi: np.ndarray) -> np.ndarray:
        """
        Verilen NumPy tensörü uzerinde model cikarimi gerceklestirir.
        """
        if not isinstance(girdi_dizisi, np.ndarray):
            raise TypeError("Girdi NumPy dizisi olmalidir.")

        if girdi_dizisi.dtype != np.float32:
            girdi_dizisi = girdi_dizisi.astype(np.float32)

        ciktilar = self.oturum.run([self.cikti_adi], {self.girdi_adi: girdi_dizisi})
        return ciktilar[0]

    def gecikme_olcumle(
        self,
        girdi_dizisi: np.ndarray,
        tekrar_sayisi: int = 100
    ) -> Dict[str, float]:
        """
        Modelin cikarim gecikmesini (p50, p95, p99, ortalama) olcer.
        """
        gecikmeler: List[float] = []

        # Isinma
        for _ in range(10):
            _ = self.tahmin_et(girdi_dizisi)

        for _ in range(tekrar_sayisi):
            t0 = time.perf_counter()
            _ = self.tahmin_et(girdi_dizisi)
            t1 = time.perf_counter()
            gecikmeler.append((t1 - t0) * 1000.0)  # ms cinsinden

        arr = np.array(gecikmeler)
        return {
            "ortalama_ms": round(float(np.mean(arr)), 3),
            "std_ms": round(float(np.std(arr)), 3),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
            "min_ms": round(float(np.min(arr)), 3),
            "maks_ms": round(float(np.max(arr)), 3),
            "fps_throughput": round(float(1000.0 / (np.mean(arr) + 1e-9) * girdi_dizisi.shape[0]), 2)
        }
