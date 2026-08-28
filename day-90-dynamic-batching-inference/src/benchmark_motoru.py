"""
Çıkarım Performansı ve Batching Benchmark Motoru
------------------------------------------------
Tekil ardışık çıkarım (B=1), statik batching ve dinamik kuyruk batching'i
gerçekçi asenkron istemci yükleri altında karşılaştıran motor.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Tuple
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
import torch.nn as nn

from .dinamik_batcher import DinamikBatchMotoru, CikarimYaniti


class BatchingBenchmarkMotoru:
    """
    Farklı batching stratejilerini gecikme, hacim ve GPU doygunluğu açısından kıyaslayan sınıf.
    """
    def __init__(self, model: nn.Module, cihaz: str = "cpu"):
        self.model = model.to(cihaz).eval()
        self.cihaz = cihaz

    def kos_ardisik_b1(self, istekler: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Her isteği tekil olarak (B=1) ardışık işler.
        """
        gecikmeler_ms = []
        toplam_istek = len(istekler)

        t0 = time.time()
        with torch.no_grad():
            for girdi in istekler:
                t_req = time.time()
                g = girdi.to(self.cihaz)
                if g.ndim == 3:
                    g = g.unsqueeze(0)
                _ = self.model(g)
                if self.cihaz == "cuda":
                    torch.cuda.synchronize()
                gecikmeler_ms.append((time.time() - t_req) * 1000.0)

        toplam_sure_sn = time.time() - t0
        throughput = toplam_istek / max(1e-5, toplam_sure_sn)

        return {
            "mod": "Sequential (B=1)",
            "toplam_istek": toplam_istek,
            "toplam_sure_sn": toplam_sure_sn,
            "throughput_req_s": throughput,
            "ortalama_gecikme_ms": float(np.mean(gecikmeler_ms)),
            "p50_gecikme_ms": float(np.percentile(gecikmeler_ms, 50)),
            "p90_gecikme_ms": float(np.percentile(gecikmeler_ms, 90)),
            "p99_gecikme_ms": float(np.percentile(gecikmeler_ms, 99)),
            "ortalama_batch": 1.0,
            "ham_gecikmeler": gecikmeler_ms
        }

    def kos_dinamik_batching(
        self,
        istekler: List[torch.Tensor],
        max_batch_size: int = 32,
        max_bekleme_ms: float = 8.0,
        es_zamanli_istemci_sayisi: int = 16
    ) -> Dict[str, Any]:
        """
        İstekleri dinamik batching motoru üzerinden eşzamanlı istemcilerle koşturur.
        """
        motor = DinamikBatchMotoru(
            model=self.model,
            max_batch_size=max_batch_size,
            max_bekleme_ms=max_bekleme_ms,
            cihaz=self.cihaz
        )

        toplam_istek = len(istekler)
        yanitlar: List[CikarimYaniti] = []

        def istemci_gonder(girdi: torch.Tensor, idx: int):
            # Gerçekçi ağ geliş varyansı (0-2ms arası küçük gecikme)
            time.sleep(random.uniform(0.0001, 0.002))
            return motor.tahmin_et_senkron(girdi, istek_id=f"req_{idx}")

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=es_zamanli_istemci_sayisi) as executor:
            future_to_req = {executor.submit(istemci_gonder, istekler[i], i): i for i in range(toplam_istek)}
            for f in as_completed(future_to_req):
                yanitlar.append(f.result())

        toplam_sure_sn = time.time() - t0
        motor.kapat()

        gecikmeler_ms = [y.toplam_gecikme_ms for y in yanitlar]
        kuyruk_ms = [y.kuyruk_suresi_ms for y in yanitlar]
        cikarim_ms = [y.cikarim_suresi_ms for y in yanitlar]
        batch_boyutlari = [y.batch_boyutu for y in yanitlar]

        throughput = toplam_istek / max(1e-5, toplam_sure_sn)

        return {
            "mod": f"Dynamic Batching (B_max={max_batch_size}, delay={max_bekleme_ms}ms)",
            "toplam_istek": toplam_istek,
            "toplam_sure_sn": toplam_sure_sn,
            "throughput_req_s": throughput,
            "ortalama_gecikme_ms": float(np.mean(gecikmeler_ms)),
            "p50_gecikme_ms": float(np.percentile(gecikmeler_ms, 50)),
            "p90_gecikme_ms": float(np.percentile(gecikmeler_ms, 90)),
            "p99_gecikme_ms": float(np.percentile(gecikmeler_ms, 99)),
            "ortalama_kuyruk_ms": float(np.mean(kuyruk_ms)),
            "ortalama_cikarim_ms": float(np.mean(cikarim_ms)),
            "ortalama_batch": float(np.mean(batch_boyutlari)),
            "ham_gecikmeler": gecikmeler_ms
        }
