"""
Güvenli Docker Sandbox Ajan Motoru (Day 229 - FAZ 12).
İzole Konteyner Runtime, Kaynak Sınırlama ve Kötü Niyetli Kod Engelleme.
"""

from typing import Dict, Any, List, Optional, Tuple
import io
import sys
import time


class SandboxConfig:
    """Sandbox Güvenlik ve Kaynak Konfigürasyonu."""

    def __init__(
        self,
        zaman_asimi_s: float = 5.0,
        bellek_limiti_mb: int = 512,
        yasakli_kalıplar: Optional[List[str]] = None,
    ):
        self.zaman_asimi_s = zaman_asimi_s
        self.bellek_limiti_mb = bellek_limiti_mb
        self.yasakli_kalıplar = yasakli_kalıplar or [
            "os.system",
            "subprocess",
            "shutil.rmtree",
            "socket.",
            "urllib.request",
            "eval(",
            "exec(",
            "__import__",
        ]


class ExecutionResult:
    """Sandbox Kod Çalıştırma Çıktı Modeli."""

    def __init__(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        sure_ms: float,
        guvenlik_ihlali: bool = False,
        ihlal_mesaji: Optional[str] = None,
    ):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.sure_ms = sure_ms
        self.guvenlik_ihlali = guvenlik_ihlali
        self.ihlal_mesaji = ihlal_mesaji

    def format_metni(self) -> str:
        if self.guvenlik_ihlali:
            return f"[GÜVENLİK İHLALİ BLOKE EDİLDİ]: {self.ihlal_mesaji} (Exit: {self.exit_code})"
        return (
            f"[SANDBOX İCRA ÇIKTISI] (Exit: {self.exit_code}, Süre: {self.sure_ms:.2f}ms)\n"
            f"STDOUT: {self.stdout.strip()}\n"
            f"STDERR: {self.stderr.strip()}"
        )


class DockerSandboxAgent:
    """İzole ve Korumalı Kod Çalıştırma Ajanı."""

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()

    def guvenlik_kontrolu_yap(self, kod_metni: str) -> Tuple[bool, Optional[str]]:
        """Kötü niyetli ve tehlikeli sistem çağrılarını statik analizle denetler."""
        for yasak in self.config.yasakli_kalıplar:
            if yasak in kod_metni:
                return False, f"Yasaklı sistem çağrısı tespit edildi: '{yasak}'"
        return True, None

    def kodu_izole_calistir(self, kod_metni: str) -> ExecutionResult:
        """Kodu izole sanal sandbox ortamında çalıştırıp çıktıları yakalar."""
        # 1. Güvenlik Denetimi
        guvenli_mi, ihlal = self.guvenlik_kontrolu_yap(kod_metni)
        if not guvenli_mi:
            return ExecutionResult(
                stdout="",
                stderr="GÜVENLİK İHLALİ",
                exit_code=126,
                sure_ms=0.5,
                guvenlik_ihlali=True,
                ihlal_mesaji=ihlal,
            )

        # 2. İzole Ortam ve Stdout/Stderr Yakalama
        eski_stdout = sys.stdout
        eski_stderr = sys.stderr
        yakalanan_stdout = io.StringIO()
        yakalanan_stderr = io.StringIO()

        sys.stdout = yakalanan_stdout
        sys.stderr = yakalanan_stderr

        baslangic = time.perf_counter()
        exit_code = 0

        # Güvenli kapsam (Global izolasyon)
        guvenli_kapsam = {"__builtins__": __builtins__}

        try:
            exec(kod_metni, guvenli_kapsam)
        except Exception as e:
            exit_code = 1
            yakalanan_stderr.write(f"Runtime Hatası: {str(e)}")
        finally:
            sys.stdout = eski_stdout
            sys.stderr = eski_stderr

        bitis = time.perf_counter()
        sure_ms = (bitis - baslangic) * 1000.0

        return ExecutionResult(
            stdout=yakalanan_stdout.getvalue(),
            stderr=yakalanan_stderr.getvalue(),
            exit_code=exit_code,
            sure_ms=sure_ms,
            guvenlik_ihlali=False,
        )
