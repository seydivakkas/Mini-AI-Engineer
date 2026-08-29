"""
Semantik Parçalama (Semantic Chunking) Motoru (Day 131 - Faz 7).
Cümle embedding'leri, ardışık kosinüs mesafesi, dinamik eşik tespiti ve semantik parça oluşturucu.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F

from .cumle_ayristirici import CumleAyristirici, BaglamTamponlayici


class SemantikParcalayici:
    """Ardışık cümleler arasındaki anlamsal mesafeyi ölçerek dinamik parçalar üreten motor."""

    def __init__(
        self,
        vektor_boyutu: int = 128,
        tampon_boyutu: int = 1,
        esik_yontemi: str = "standart_sapma",  # "standart_sapma", "yuzdelik_dilim", "sabit"
        esik_katsayisi: float = 0.75,
        sabit_esik: float = 0.40,
        yuzdelik_dilim: float = 80.0,
    ):
        self.vektor_boyutu = vektor_boyutu
        self.tampon_boyutu = tampon_boyutu
        self.esik_yontemi = esik_yontemi
        self.esik_katsayisi = esik_katsayisi
        self.sabit_esik = sabit_esik
        self.yuzdelik_dilim = yuzdelik_dilim

    def _metin_vektorlestir(self, metinler: List[str]) -> torch.Tensor:
        """
        Metinleri tutarlı, deterministik ve anlamsal semantik vektörlere dönüştürür.
        PyTorch tensörleri ile normalize edilmiş L2 embedding üretir.
        """
        vektorler = []
        for metin in metinler:
            # Deterministik tohum ile kelime öbeklerine göre zengin embedding
            sozcukler = metin.lower().split()
            vektor = np.zeros(self.vektor_boyutu, dtype=np.float32)

            for idx, kelime in enumerate(sozcukler):
                np.random.seed(abs(hash(kelime)) % (2**31))
                agirlik = 1.0 / (idx + 1) ** 0.5
                vektor += np.random.randn(self.vektor_boyutu).astype(np.float32) * agirlik

            vektorler.append(vektor)

        tensör = torch.tensor(np.array(vektorler), dtype=torch.float32)
        return F.normalize(tensör, p=2, dim=1)

    def kosinus_mesafesi_hesapla(self, embeddingler: torch.Tensor) -> List[float]:
        """Ardışık cümle vektörleri arasındaki kosinüs mesafesini (1 - CosSim) hesaplar."""
        n = embeddingler.shape[0]
        if n <= 1:
            return []

        mesafeler = []
        for i in range(n - 1):
            v1 = embeddingler[i : i + 1]
            v2 = embeddingler[i + 1 : i + 2]
            benzerlik = F.cosine_similarity(v1, v2).item()
            mesafe = float(np.clip(1.0 - benzerlik, 0.0, 2.0))
            mesafeler.append(mesafe)

        return mesafeler

    def esik_degeri_belirle(self, mesafeler: List[float]) -> float:
        """Kosinüs mesafeleri dizisi üzerinden semantik kırılma eşik değerini belirler."""
        if not mesafeler:
            return self.sabit_esik

        dizi = np.array(mesafeler)
        if self.esik_yontemi == "standart_sapma":
            ortalama = float(np.mean(dizi))
            std = float(np.std(dizi))
            return ortalama + (self.esik_katsayisi * std)
        elif self.esik_yontemi == "yuzdelik_dilim":
            return float(np.percentile(dizi, self.yuzdelik_dilim))
        else:
            return self.sabit_esik

    def parcala(self, ham_metin: str) -> Dict[str, Any]:
        """
        Ham metni cümlelere ayırır, tamponlar, embedding çıkarır ve semantik parçalara böler.
        """
        # 1. Cümlelere Ayrıştır
        cumleler = CumleAyristirici.ayristir(ham_metin)
        if not cumleler:
            return {
                "toplam_cumle": 0,
                "toplam_parca": 0,
                "parcalar": [],
                "mesafeler": [],
                "esik": 0.0,
            }

        # 2. Bağlam Tamponu Oluştur
        tamponlu_listesi = BaglamTamponlayici.tampon_olustur(cumleler, self.tampon_boyutu)
        baglam_metinleri = [item["birlestirilmis_baglam"] for item in tamponlu_listesi]

        # 3. Embedding Üret ve Kosinüs Mesafesi Hesapla
        embeddingler = self._metin_vektorlestir(baglam_metinleri)
        mesafeler = self.kosinus_mesafesi_hesapla(embeddingler)

        # 4. Dinamik Eşik Belirle
        esik = self.esik_degeri_belirle(mesafeler)

        # 5. Kırılma Noktalarına Göre Parçala
        parcalar: List[Dict[str, Any]] = []
        mevcut_parca_cumleleri: List[str] = [cumleler[0]]
        baslangic_idx = 0

        for i, mesafe in enumerate(mesafeler):
            if mesafe > esik:
                # Kırılma noktası: Mevcut parçayı kaydet ve yeni parça başlat
                parca_metni = " ".join(mevcut_parca_cumleleri)
                parcalar.append({
                    "parca_id": f"CHUNK_{len(parcalar)+1:03d}",
                    "metin": parca_metni,
                    "cumle_sayisi": len(mevcut_parca_cumleleri),
                    "karakter_sayisi": len(parca_metni),
                    "baslangic_cumle_idx": baslangic_idx,
                    "bitis_cumle_idx": i,
                })
                mevcut_parca_cumleleri = [cumleler[i + 1]]
                baslangic_idx = i + 1
            else:
                mevcut_parca_cumleleri.append(cumleler[i + 1])

        # Son parçayı ekle
        if mevcut_parca_cumleleri:
            parca_metni = " ".join(mevcut_parca_cumleleri)
            parcalar.append({
                "parca_id": f"CHUNK_{len(parcalar)+1:03d}",
                "metin": parca_metni,
                "cumle_sayisi": len(mevcut_parca_cumleleri),
                "karakter_sayisi": len(parca_metni),
                "baslangic_cumle_idx": baslangic_idx,
                "bitis_cumle_idx": len(cumleler) - 1,
            })

        return {
            "toplam_cumle": len(cumleler),
            "toplam_parca": len(parcalar),
            "parcalar": parcalar,
            "mesafeler": mesafeler,
            "esik": esik,
        }
