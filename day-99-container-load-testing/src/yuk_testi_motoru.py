"""
Eşzamanlı Yük ve Stres Testi Motoru (Day 99).
Asenkron HTTP istemcileriyle basamaklı (ramping) yük simülasyonu ve SLA doğrulaması yapar.
"""

import io
import time
import base64
import asyncio
from typing import List, Dict, Any, Tuple
from PIL import Image
import numpy as np
import httpx


class YukTestiMotoru:
    """Eşzamanlı kullanıcı seviyelerinde yük simülasyonu yapan motor."""

    def __init__(self, app, base_url: str = "http://testserver"):
        self.app = app
        self.base_url = base_url

        # Sentetik görsel hazırla
        img = Image.new("RGB", (32, 32), color=(45, 90, 180))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self.img_bytes = buf.getvalue()
        self.b64_str = base64.b64encode(self.img_bytes).decode("utf-8")

    async def _tekil_istek(self, client: httpx.AsyncClient, istek_turu: str = "predict") -> Tuple[float, int, str]:
        """Tek bir HTTP isteği gönderir ve gecikme, durum kodu ve endpoint adını döner."""
        t0 = time.perf_counter()
        try:
            if istek_turu == "predict":
                files = {"file": ("test.png", self.img_bytes, "image/png")}
                resp = await client.post("/predict?top_k=5", files=files)
            elif istek_turu == "base64":
                payload = {"base64_goruntu": self.b64_str, "top_k": 3}
                resp = await client.post("/predict/base64", json=payload)
            else:
                resp = await client.get("/health")
            code = resp.status_code
        except Exception:
            code = 500

        t1 = time.perf_counter()
        gecikme_ms = (t1 - t0) * 1000.0
        return gecikme_ms, code, istek_turu

    async def eszamanli_seviye_testi(
        self,
        kullanici_sayisi: int,
        kullanici_basina_istek: int = 5,
    ) -> Dict[str, Any]:
        """Belirtilen eşzamanlı kullanıcı sayısı ile yük testi koşturur."""
        toplam_istek = kullanici_sayisi * kullanici_basina_istek
        istek_turleri = []
        for i in range(toplam_istek):
            r = i % 10
            if r < 6:
                istek_turleri.append("predict")
            elif r < 9:
                istek_turleri.append("base64")
            else:
                istek_turleri.append("health")

        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(transport=transport, base_url=self.base_url) as client:
            t0 = time.perf_counter()
            gorevler = [self._tekil_istek(client, tur) for tur in istek_turleri]
            sonuclar = await asyncio.gather(*gorevler)
            toplam_sure_sn = time.perf_counter() - t0

        gecikmeler = [s[0] for s in sonuclar]
        kodlar = [s[1] for s in sonuclar]
        basarili_sayisi = sum(1 for c in kodlar if c == 200)
        hata_sayisi = len(kodlar) - basarili_sayisi
        hata_orani = (hata_sayisi / len(kodlar)) * 100.0 if kodlar else 0.0

        p50 = float(np.percentile(gecikmeler, 50))
        p90 = float(np.percentile(gecikmeler, 90))
        p99 = float(np.percentile(gecikmeler, 99))
        throughput_rps = len(kodlar) / max(toplam_sure_sn, 1e-4)

        return {
            "kullanici_sayisi": kullanici_sayisi,
            "toplam_istek": len(kodlar),
            "basarili_sayisi": basarili_sayisi,
            "hata_sayisi": hata_sayisi,
            "hata_orani_yuzde": round(hata_orani, 2),
            "throughput_rps": round(throughput_rps, 1),
            "p50_ms": round(p50, 2),
            "p90_ms": round(p90, 2),
            "p99_ms": round(p99, 2),
            "ortalama_ms": round(float(np.mean(gecikmeler)), 2),
            "gecikmeler": gecikmeler,
        }

    async def basamakli_yuk_testi(
        self,
        kullanici_basamaklari: List[int] = [1, 5, 10, 25, 50, 100],
    ) -> List[Dict[str, Any]]:
        """Farklı eşzamanlı kullanıcı basamaklarında stres testi koşturur."""
        tum_sonuclar = []
        for k in kullanici_basamaklari:
            res = await self.eszamanli_seviye_testi(k, kullanici_basina_istek=4)
            tum_sonuclar.append(res)
        return tum_sonuclar
