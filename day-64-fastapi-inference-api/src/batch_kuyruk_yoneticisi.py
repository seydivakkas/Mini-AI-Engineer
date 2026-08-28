"""
Asenkron Dinamik Batch Kuyruk Yöneticisi (Dynamic Batching Queue Manager).
"""

import asyncio
import time
from typing import Dict, Any, List, Tuple
from .model_motoru import YapayZekaModelMotoru


class DinamikBatchKuyrugu:
    """Tekil gelen istekleri zaman penceresi ve batch boyutuna göre birleştirip toplu işleyen asenkron kuyruk."""

    def __init__(
        self,
        model_motoru: YapayZekaModelMotoru,
        maks_batch_boyutu: int = 16,
        maks_bekleme_ms: float = 10.0
    ):
        self.model_motoru = model_motoru
        self.maks_batch_boyutu = maks_batch_boyutu
        self.maks_bekleme_s = maks_bekleme_ms / 1000.0
        self.kuyruk: asyncio.Queue = asyncio.Queue()
        self._calisiyor = False
        self._gorev: asyncio.Task = None

    def baslat(self) -> None:
        """Kuyruk tüketici arka plan görevini (Worker) başlatır."""
        if not self._calisiyor:
            self._calisiyor = True
            self._gorev = asyncio.create_task(self._kuyruk_tuketici())

    async def durdur(self) -> None:
        """Kuyruğu nazikçe durdurur."""
        self._calisiyor = False
        if self._gorev:
            self._gorev.cancel()
            try:
                await self._gorev
            except asyncio.CancelledError:
                pass

    async def tahmin_kuyruga_ekle(self, istek: Dict[str, Any]) -> Dict[str, Any]:
        """Tekil isteği kuyruğa ekler ve batch işlenene kadar asenkron olarak bekler."""
        loop = asyncio.get_running_loop()
        gelecek_sonuc = loop.create_future()
        await self.kuyruk.put((istek, gelecek_sonuc))
        return await gelecek_sonuc

    async def _kuyruk_tuketici(self) -> None:
        """Kuyruktan gelen istekleri toplayıp batch olarak modele gönderen döngü."""
        while self._calisiyor:
            try:
                # İlk elemanı bekle
                ilk_istek, ilk_future = await self.kuyruk.get()
                batch_istekler = [ilk_istek]
                batch_futures = [ilk_future]

                baslangic_zamani = time.perf_counter()

                # Maksimum bekleme süresi veya maks_batch_boyutuna kadar topla
                while len(batch_istekler) < self.maks_batch_boyutu:
                    kalan_sure = self.maks_bekleme_s - (time.perf_counter() - baslangic_zamani)
                    if kalan_sure <= 0:
                        break
                    try:
                        ek_istek, ek_future = await asyncio.wait_for(self.kuyruk.get(), timeout=kalan_sure)
                        batch_istekler.append(ek_istek)
                        batch_futures.append(ek_future)
                    except asyncio.TimeoutError:
                        break

                # Toplu Çıkarımı Çalıştır
                sonuclar = self.model_motoru.toplu_tahmin(batch_istekler)

                # Gelecek nesnelerine (Futures) sonuçları dağıt
                for fut, res in zip(batch_futures, sonuclar):
                    if not fut.cancelled():
                        fut.set_result(res)

            except asyncio.CancelledError:
                break
            except Exception as e:
                for fut in batch_futures:
                    if not fut.cancelled():
                        fut.set_exception(e)
