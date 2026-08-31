"""
Tesla TPM ve Secure Boot Profilleyici Modülü
============================================
Bu modül; SHA-256 hash hesaplama, sabit zamanlı imza doğrulama
ve 4 aşamalı Güven Zinciri (Chain of Trust) çözüm hızını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_secure_boot_dogrulayici import TeslaSecureBootValidator


class TeslaTPMProfilleyici:
    """
    Tesla TPM Secure Boot Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_secure_boot(self) -> Dict[str, Any]:
        validator = TeslaSecureBootValidator()

        gecikmeler_us: List[float] = []
        ciktilar = None

        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            ciktilar = validator.validate_full_secure_boot_chain(simulate_tamper=False)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "validation_step_ortalama_us": t_avg_us,
            "validation_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_dogrulama_kapasitesi": int(1e6 / max(t_avg_us, 1e-4)),
            "chain_verified": ciktilar["chain_verified"],
            "status_text": ciktilar["status_text"],
            "stages": [
                ciktilar["stage1_rom_rot"],
                ciktilar["stage2_uboot_sig"],
                ciktilar["stage3_kernel_sig"],
                ciktilar["stage4_dm_verity"]
            ],
            "hashes": {
                "ROM": ciktilar["rom_hash"][:16] + "...",
                "U-Boot": ciktilar["uboot_hash"][:16] + "...",
                "Kernel": ciktilar["kernel_hash"][:16] + "...",
                "Rootfs": ciktilar["rootfs_hash"][:16] + "..."
            },
            "gecikmeler": gecikmeler_us[:200]
        }
