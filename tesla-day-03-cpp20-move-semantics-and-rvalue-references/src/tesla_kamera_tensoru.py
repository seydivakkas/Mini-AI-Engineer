"""
Tesla Kamera Tensoru ve Tasima Semantigi (Tesla C++20 Move Semantics & Zero-Copy)
==================================================================================
Bu modul, Tesla FSD 8-kamera 4K/FHD yuksek cozunurluklu goruntu tensörlerinin
kopyalamasiz (Zero-Copy) olarak Rvalue Referanslari ($&&$) ve Move Semantics ile
NPU/GPU islem hatlarina aktarilmasini gercekler.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

from typing import Optional, Tuple, Dict, Any, List
import time
import numpy as np


class TeslaKameraTensoru:
    """
    Tesla FSD 8-Kamera Ham Goruntu ve Feature Tensor Sinifi.
    C++20 Rule of Five (Kopyalama Yapici, Tasima Yapici, Kopyalama Atama, Tasima Atama, Yikici)
    semantigini ve sifir-kopyalama (zero-copy) mekanizmasini uygular.
    """
    def __init__(self, kamera_adi: str, genislik: int = 1920, yukseklik: int = 1080, kanal: int = 3, veri_tamponu: Optional[bytearray] = None):
        self.kamera_adi = kamera_adi
        self.genislik = genislik
        self.yukseklik = yukseklik
        self.kanal = kanal
        self.boyut_bayt = genislik * yukseklik * kanal
        self.zaman_damgasi_ns = time.time_ns()
        
        # Eger hazir tampon verildiyse sahiplen, yoksa tahsis et
        if veri_tamponu is not None:
            self._tampon: Optional[bytearray] = veri_tamponu
        else:
            self._tampon = bytearray(self.boyut_bayt)
            
        self._gecerli = True
        self._bellek_id = id(self._tampon)

    @property
    def gecerli_mi(self) -> bool:
        return self._gecerli and self._tampon is not None

    @property
    def boyut_mb(self) -> float:
        return self.boyut_bayt / (1024 * 1024)

    @property
    def bellek_adresi(self) -> int:
        return self._bellek_id if self.gecerli_mi else 0

    def derin_kopyala(self) -> 'TeslaKameraTensoru':
        """
        C++ Copy Constructor: Tamponu baştan sona yeniden kopyalar (O(N) - Pahalı).
        """
        if not self.gecerli_mi or self._tampon is None:
            raise RuntimeError("HATA: Gecersiz veya tasinmis (moved-from) tensorden kopyalama yapilamaz!")
            
        yeni_tampon = bytearray(self._tampon)  # 6.2 MB kopyalama maliyeti
        yeni_tensor = TeslaKameraTensoru(
            kamera_adi=f"{self.kamera_adi}_kopya",
            genislik=self.genislik,
            yukseklik=self.yukseklik,
            kanal=self.kanal,
            veri_tamponu=yeni_tampon
        )
        return yeni_tensor

    def tasi(self) -> 'TeslaKameraTensoru':
        """
        C++20 Move Constructor (std::move): Sahipligi O(1) surede yeni tensore aktarir.
        Eski tensorun bellek baglantisi koparilir (Moved-from state).
        """
        if not self.gecerli_mi or self._tampon is None:
            raise RuntimeError("HATA: Gecersiz veya tasinmis (moved-from) tensor tasinamaz!")

        # Sahiplik aktarimi (Pointer transfer)
        aktarilan_tampon = self._tampon
        aktarilan_id = self._bellek_id
        
        # Eski tensorun kaynaklarini sifirla
        self._tampon = None
        self._gecerli = False
        self._bellek_id = 0

        hedef_tensor = TeslaKameraTensoru(
            kamera_adi=self.kamera_adi,
            genislik=self.genislik,
            yukseklik=self.yukseklik,
            kanal=self.kanal,
            veri_tamponu=aktarilan_tampon
        )
        hedef_tensor._bellek_id = aktarilan_id  # Bellek adresi ayni kalir (Sifir Kopyalama!)
        return hedef_tensor

    def tasima_ile_ata(self, diger: 'TeslaKameraTensoru'):
        """
        C++20 Move Assignment Operator (operator=(&&)): Mevcut kaynagi temizler, yenisini sahiplenir.
        """
        if self is diger:
            return  # Self-assignment korumasi
            
        if not diger.gecerli_mi or diger._tampon is None:
            raise RuntimeError("HATA: Gecersiz tensorden tasima atamasi yapilamaz!")

        # Mevcut tamponu serbest birak
        self._tampon = diger._tampon
        self._bellek_id = diger._bellek_id
        self.kamera_adi = diger.kamera_adi
        self.genislik = diger.genislik
        self.yukseklik = diger.yukseklik
        self.kanal = diger.kanal
        self.boyut_bayt = diger.boyut_bayt
        self._gecerli = True

        # Kaynak nesneyi bosalt
        diger._tampon = None
        diger._gecerli = False
        diger._bellek_id = 0


class TeslaFSDKameraHatti:
    """
    Tesla FSD 8 Kamera Akisini (Surround Vision) Move Semantics ile Yoneten Pipeline.
    """
    KAMERA_LISTESI = [
        "on_merkez_ana", "on_genis_aci", "on_dar_aci",
        "sol_on_yan", "sag_on_yan", "sol_arka_yan", "sag_arka_yan", "arka_merkez"
    ]

    def __init__(self):
        self.islenen_kare_sayisi = 0
        self.toplam_tasinan_mb = 0.0

    def kamera_kare_uret(self, kamera_adi: str) -> TeslaKameraTensoru:
        return TeslaKameraTensoru(kamera_adi=kamera_adi, genislik=1920, yukseklik=1080, kanal=3)

    def npu_girisine_tasi(self, tensor: TeslaKameraTensoru) -> Tuple[TeslaKameraTensoru, float]:
        """
        Kamera tensörünü sıfır kopyalama ile NPU tensör motoruna taşır.
        """
        t0 = time.perf_counter_ns()
        npu_tensör = tensor.tasi()
        t1 = time.perf_counter_ns()
        
        gecikme_ns = float(t1 - t0)
        self.islenen_kare_sayisi += 1
        self.toplam_tasinan_mb += npu_tensör.boyut_mb
        return npu_tensör, gecikme_ns
