"""
Tesla RAII Profilleyici ve Sizinti Analiz Benchmark'i
=====================================================
Bu modul, RAII ve Custom Deleter yaklasiminin ham isaretcilere (Raw Pointers)
kiyosla istisna guvenligi (Exception Safety), kaynak sizintisi (Resource Leak)
ve yok etme gecikmelerini nanosecond seviyesinde analiz eder.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import time
import numpy as np
from typing import Dict, List, Any
from src.tesla_raii_kaynak_yoneticisi import (
    TeslaDonanimKaynagi,
    DonanimKaynakTipi,
    TeslaCANSoketRAII,
    OzelSiliciAkilliIsaretci,
    TeslaKaynakIzlemeMerkezi
)


class TeslaRAIIProfilleyici:
    """
    RAII ve Akilli Isaretcilerin otomotiv guvenligi (ASIL-D) performans profilleyicisi.
    """
    def __init__(self, dongu_sayisi: int = 1000):
        self.dongu_sayisi = dongu_sayisi

    def benchmark_istisna_guvenligi_ve_sizinti(self) -> Dict[str, Any]:
        """
        Istisna firlatma durumunda RAII vs Ham Isaretci sizinti karsilastirmasi.
        """
        merkez_raii = TeslaKaynakIzlemeMerkezi()
        merkez_ham = TeslaKaynakIzlemeMerkezi()
        
        # 1. RAII Testi (Hata olussa bile %100 temizlik garantisi)
        raii_gecikmeleri_ns: List[float] = []
        for i in range(self.dongu_sayisi):
            t0 = time.perf_counter_ns()
            try:
                with TeslaCANSoketRAII(arayuz_adi=f"can_raii_{i}") as soket:
                    merkez_raii.kaydet(soket.kaynak)
                    soket.telemetri_yaz(0x100, b"TEST_DATA")
                    if i % 5 == 0:
                        raise ValueError("Simule Edilmis CAN Hatti Hatasi!")
            except ValueError:
                pass
            t1 = time.perf_counter_ns()
            raii_gecikmeleri_ns.append(float(t1 - t0))

        # 2. Ham Isaretci (Raw Pointer) Testi (Istisna aninda manuel kapatma atlanir)
        ham_gecikmeleri_ns: List[float] = []
        for i in range(self.dongu_sayisi):
            t0 = time.perf_counter_ns()
            kaynak = TeslaDonanimKaynagi(
                kaynak_id=f"SOCKET_raw_{i}",
                tip=DonanimKaynakTipi.CAN_SOKET,
                aciklayici_no=i
            )
            merkez_ham.kaydet(kaynak)
            try:
                kaynak.veri_gonder(b"TEST_DATA")
                if i % 5 == 0:
                    raise ValueError("Simule Edilmis CAN Hatti Hatasi!")
                kaynak.donanim_kapat()  # Sadece basarili olursa kapanir
            except ValueError:
                pass  # Hata durumunda manuel kapatma unutuldu/atlandi!
            t1 = time.perf_counter_ns()
            ham_gecikmeleri_ns.append(float(t1 - t0))

        raii_sizinti_sayisi = merkez_raii.aktif_acik_kaynak_sayisi()
        ham_sizinti_sayisi = merkez_ham.aktif_acik_kaynak_sayisi()

        raii_dizi = np.array(raii_gecikmeleri_ns)
        ham_dizi = np.array(ham_gecikmeleri_ns)

        return {
            "toplam_islem": self.dongu_sayisi,
            "raii_sizinti_sayisi": raii_sizinti_sayisi,
            "raii_sizinti_orani": float(merkez_raii.sizinti_orani()),
            "ham_sizinti_sayisi": ham_sizinti_sayisi,
            "ham_sizinti_orani": float(merkez_ham.sizinti_orani()),
            "raii_ortalama_ns": float(np.mean(raii_dizi)),
            "raii_p99_ns": float(np.percentile(raii_dizi, 99)),
            "raii_jitter_ns": float(np.std(raii_dizi)),
            "ham_ortalama_ns": float(np.mean(ham_dizi)),
            "ham_p99_ns": float(np.percentile(ham_dizi, 99)),
            "ham_jitter_ns": float(np.std(ham_dizi)),
            "raii_gecikmeleri": raii_gecikmeleri_ns,
            "ham_gecikmeleri": ham_gecikmeleri_ns,
            "guvenlik_skoru": float((1.0 - merkez_raii.sizinti_orani()) * 100.0)
        }

    def benchmark_custom_deleter_turleri(self) -> Dict[str, float]:
        """
        Durumsuz Lambda vs std::function vs Sanal Yok Edici (Virtual Destructor) gecikme analizi.
        """
        # 1. Stateless Lambda (Sifir ek yuk - Zero Overhead)
        t0 = time.perf_counter_ns()
        for _ in range(5000):
            k = TeslaDonanimKaynagi("K1", DonanimKaynakTipi.GPU_TAMPON)
            ptr = OzelSiliciAkilliIsaretci(k, lambda obj: obj.donanim_kapat())
            ptr.serbest_birak_ve_yok_et()
        t1 = time.perf_counter_ns()
        lambda_sure_ns = (t1 - t0) / 5000.0

        # 2. std::function / Dynamic Callable (Hafif dynamic dispatch maliyeti)
        class DinamikSilici:
            def __call__(self, obj):
                obj.donanim_kapat()

        dinamik_obj = DinamikSilici()
        t0 = time.perf_counter_ns()
        for _ in range(5000):
            k = TeslaDonanimKaynagi("K2", DonanimKaynakTipi.GPU_TAMPON)
            ptr = OzelSiliciAkilliIsaretci(k, dinamik_obj)
            ptr.serbest_birak_ve_yok_et()
        t1 = time.perf_counter_ns()
        function_sure_ns = (t1 - t0) / 5000.0

        return {
            "stateless_lambda_ns": float(lambda_sure_ns),
            "dynamic_function_ns": float(function_sure_ns),
            "hiz_farki_yuzde": float(((function_sure_ns - lambda_sure_ns) / max(lambda_sure_ns, 1e-6)) * 100.0)
        }
