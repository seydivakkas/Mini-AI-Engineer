"""
Kuyruk Tabanlı Dinamik Batching Çıkarım Motoru (Dynamic Batching Engine)
-----------------------------------------------------------------------
Triton Inference Server ve vLLM mimarisi standartlarında asenkron istemci
isteklerini kuyrukta toplayıp optimum GPU batch tensörlerine dönüştüren motor.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time
import queue
import threading
from concurrent.futures import Future
import torch
import torch.nn as nn


@dataclass
class CikarimIstegi:
    """Tekil istemciden gelen çıkarım isteği nesnesi."""
    istek_id: str
    girdi: torch.Tensor
    varis_zamani: float
    gelecek_yanit: Future


@dataclass
class CikarimYaniti:
    """İstemciye dönen sonuç ve ayrıntılı gecikme profili."""
    istek_id: str
    cikis: torch.Tensor
    kuyruk_suresi_ms: float
    cikarim_suresi_ms: float
    toplam_gecikme_ms: float
    batch_boyutu: int


class DinamikBatchMotoru:
    """
    Asenkron istekleri toplayarak GPU için dinamik mikro-batch oluşturan çıkarım motoru.
    """
    def __init__(
        self,
        model: nn.Module,
        max_batch_size: int = 32,
        max_bekleme_ms: float = 8.0,
        cihaz: str = "cpu"
    ):
        self.model = model.to(cihaz).eval()
        self.max_batch_size = max_batch_size
        self.max_bekleme_sn = max_bekleme_ms / 1000.0
        self.cihaz = cihaz

        self.kuyruk: queue.Queue[CikarimIstegi] = queue.Queue()
        self.calisiyor = True

        # İstatistikler
        self.toplam_istek = 0
        self.toplam_batch = 0
        self.toplam_islem_suresi_sn = 0.0

        # Arka plan çıkarım iş parçacığı
        self.is_parcacigi = threading.Thread(target=self._arka_plan_isleyici, daemon=True)
        self.is_parcacigi.start()

    def tahmin_et_asenkron(self, girdi: torch.Tensor, istek_id: Optional[str] = None) -> Future:
        """
        Model çıkarımı için istek kuyruğuna girdi ekler ve bir Future nesnesi döndürür.
        """
        if not self.calisiyor:
            raise RuntimeError("DinamikBatchMotoru kapatılmış durumda, yeni istek kabul edilemez!")

        if istek_id is None:
            istek_id = f"req_{time.time_ns()}"

        # Girdinin 4D [1, C, H, W] veya 3D [C, H, W] olduğundan emin ol
        if girdi.ndim == 3:
            girdi = girdi.unsqueeze(0)
        elif girdi.ndim == 4 and girdi.size(0) != 1:
            raise ValueError(f"Tekil istek batch boyutu 1 olmalıdır. Alınan: {girdi.shape}")

        gelecek = Future()
        istek = CikarimIstegi(
            istek_id=istek_id,
            girdi=girdi.cpu(),  # Kuyrukta CPU'da beklet, GPU belleğini şişirme
            varis_zamani=time.time(),
            gelecek_yanit=gelecek
        )

        self.kuyruk.put(istek)
        return gelecek

    def tahmin_et_senkron(self, girdi: torch.Tensor, istek_id: Optional[str] = None, zaman_asimi_sn: float = 5.0) -> CikarimYaniti:
        """
        Senkron olarak sonucun hesaplanmasını bekler.
        """
        gelecek = self.tahmin_et_asenkron(girdi, istek_id)
        return gelecek.result(timeout=zaman_asimi_sn)

    def _arka_plan_isleyici(self) -> None:
        """
        Kuyruğu dinleyerek max_batch_size veya max_bekleme_sn sınırına göre batch oluşturan döngü.
        """
        while self.calisiyor:
            try:
                # İlk isteği bekle
                ilk_istek = self.kuyruk.get(timeout=0.05)
            except queue.Empty:
                continue

            batch_listesi: List[CikarimIstegi] = [ilk_istek]
            toplama_baslangic = time.time()

            # Max batch dolana veya bekleme süresi aşılana kadar ek istekleri topla
            while len(batch_listesi) < self.max_batch_size:
                gecen_sure = time.time() - toplama_baslangic
                kalan_sure = self.max_bekleme_sn - gecen_sure

                if kalan_sure <= 0:
                    break

                try:
                    ek_istek = self.kuyruk.get(timeout=max(0.0001, kalan_sure))
                    batch_listesi.append(ek_istek)
                except queue.Empty:
                    break

            # Batch'i GPU'da işlet
            self._batch_isle(batch_listesi)

    def _batch_isle(self, batch_listesi: List[CikarimIstegi]) -> None:
        """
        Toplanan istekleri birleştirip tek bir GPU forward pass'inde koşturur ve sonuçları dağıtır.
        """
        if not batch_listesi:
            return

        b_boyutu = len(batch_listesi)
        cikarim_baslangic = time.time()

        try:
            # 1. İstekleri tek bir tensörde birleştir [B, C, H, W]
            toplu_girdi = torch.cat([ist.girdi for ist in batch_listesi], dim=0).to(self.cihaz)

            # 2. Tekil GPU Forward Pass
            with torch.no_grad():
                toplu_cikis = self.model(toplu_girdi)
                if self.cihaz == "cuda":
                    torch.cuda.synchronize()

            cikarim_bitis = time.time()
            cikarim_suresi_ms = (cikarim_bitis - cikarim_baslangic) * 1000.0

            # 3. Çıktıları dilimle (slice) ve her istemcinin Future nesnesine ata
            for i, ist in enumerate(batch_listesi):
                kuyruk_suresi_ms = (cikarim_baslangic - ist.varis_zamani) * 1000.0
                toplam_gecikme_ms = (cikarim_bitis - ist.varis_zamani) * 1000.0

                tekil_cikis = toplu_cikis[i:i+1].cpu()
                yanit = CikarimYaniti(
                    istek_id=ist.istek_id,
                    cikis=tekil_cikis,
                    kuyruk_suresi_ms=kuyruk_suresi_ms,
                    cikarim_suresi_ms=cikarim_suresi_ms,
                    toplam_gecikme_ms=toplam_gecikme_ms,
                    batch_boyutu=b_boyutu
                )
                ist.gelecek_yanit.set_result(yanit)

            self.toplam_istek += b_boyutu
            self.toplam_batch += 1

        except Exception as e:
            for ist in batch_listesi:
                if not ist.gelecek_yanit.done():
                    ist.gelecek_yanit.set_exception(e)

    def kapat(self) -> None:
        """
        Motoru güvenli şekilde durdurur.
        """
        self.calisiyor = False
        if self.is_parcacigi.is_alive():
            self.is_parcacigi.join(timeout=1.0)
