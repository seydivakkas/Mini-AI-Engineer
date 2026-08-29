"""
İzole Kod Çalıştırıcı ve Grafik Yakalayıcı Modülü (Day 125 - Faz 7).
Kısıtlı globals/locals sözlüğü, stdio yönlendirmesi ve otomatik matplotlib çizim yakalama motoru.
"""

import sys
import io
import time
import traceback
import math
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI olmadan arka planda çizim
import matplotlib.pyplot as plt

from .guvenlik_denetleyicisi import AstGuvenlikDenetleyicisi


class CalismaSonucu:
    """İzole kod çalıştırma sonucunu temsil eden veri sınıfı."""

    def __init__(
        self,
        basarili: bool,
        stdout: str,
        stderr: str,
        calisma_suresi_ms: float,
        grafik_sayisi: int = 0,
        grafik_dosyalari: Optional[List[str]] = None,
        guvenlik_ihlalleri: Optional[List[str]] = None,
    ):
        self.basarili = basarili
        self.stdout = stdout
        self.stderr = stderr
        self.calisma_suresi_ms = calisma_suresi_ms
        self.grafik_sayisi = grafik_sayisi
        self.grafik_dosyalari = grafik_dosyalari or []
        self.guvenlik_ihlalleri = guvenlik_ihlalleri or []


class IzoleKodCalistirici:
    """Güvenli kısıtlı ortamda Python kodunu çalıştıran ve çıktıları toplayan motor."""

    def __init__(self):
        self.guvenli_builtins = {
            "print": print,
            "range": range,
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "enumerate": enumerate,
            "zip": zip,
            "isinstance": isinstance,
            "type": type,
            "sorted": sorted,
            "reversed": reversed,
            "map": map,
            "filter": filter,
            "all": all,
            "any": any,
        }

    def _olustur_guvenli_ortam(self) -> Dict[str, Any]:
        """İzin verilen modülleri ve fonksiyonları içeren izole çalışma ortamı hazırlar."""
        return {
            "__builtins__": self.guvenli_builtins,
            "np": np,
            "numpy": np,
            "math": math,
            "plt": plt,
            "matplotlib": matplotlib,
        }

    def calistir(self, kod_metni: str, grafik_kayit_dizini: Optional[str] = None) -> CalismaSonucu:
        """
        Kodu önce AST denetiminden geçirir, güvenliyse izole ortamda çalıştırır.
        """
        # 1. AST Statik Güvenlik Denetimi
        guvenli_mi, ihlaller, _ = AstGuvenlikDenetleyicisi.denetle(kod_metni)
        if not guvenli_mi:
            return CalismaSonucu(
                basarili=False,
                stdout="",
                stderr="GÜVENLİK İHLALİ: Kod yürütmesi engellendi.",
                calisma_suresi_ms=0.0,
                grafik_sayisi=0,
                guvenlik_ihlalleri=ihlaller,
            )

        # 2. Ortam Hazırlığı ve Stdio Yönlendirmesi
        ortam = self._olustur_guvenli_ortam()
        eski_stdout = sys.stdout
        eski_stderr = sys.stderr
        yonlendirilmis_stdout = io.StringIO()
        yonlendirilmis_stderr = io.StringIO()

        plt.close("all")  # Önceki çizimleri temizle
        baslangic_zamani = time.perf_counter()
        basarili = True
        hata_metni = ""
        uretilen_grafikler: List[str] = []

        try:
            sys.stdout = yonlendirilmis_stdout
            sys.stderr = yonlendirilmis_stderr

            # Kod İcrası
            exec(kod_metni, ortam)

            # Çizilen Matplotlib Figürlerini Yakala
            fig_sayisi = plt.gcf().number if plt.get_fignums() else 0
            if fig_sayisi > 0 and grafik_kayit_dizini:
                import os
                os.makedirs(grafik_kayit_dizini, exist_ok=True)
                for fignum in plt.get_fignums():
                    fig = plt.figure(fignum)
                    dosya_adi = os.path.join(grafik_kayit_dizini, f"fig_{fignum}.png")
                    fig.savefig(dosya_adi, dpi=200, bbox_inches="tight")
                    uretilen_grafikler.append(dosya_adi)

        except Exception as e:
            basarili = False
            hata_metni = traceback.format_exc()
        finally:
            sys.stdout = eski_stdout
            sys.stderr = eski_stderr
            bitis_zamani = time.perf_counter()

        calisma_suresi = (bitis_zamani - baslangic_zamani) * 1000.0
        cikan_stdout = yonlendirilmis_stdout.getvalue()
        cikan_stderr = yonlendirilmis_stderr.getvalue()
        if hata_metni:
            cikan_stderr = (cikan_stderr + "\n" + hata_metni).strip()

        return CalismaSonucu(
            basarili=basarili,
            stdout=cikan_stdout,
            stderr=cikan_stderr,
            calisma_suresi_ms=calisma_suresi,
            grafik_sayisi=len(plt.get_fignums()),
            grafik_dosyalari=uretilen_grafikler,
            guvenlik_ihlalleri=[],
        )
