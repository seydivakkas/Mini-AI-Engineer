"""
Tesla Sandbox Profilleyici Modülü
=================================
Bu modül; Seccomp-BPF sistem çağrısı filtreleme gecikmesini ve
saniyelik çağrı inceleme kapasitesini profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_chromium_sandbox_seccomp import TeslaChromiumSeccompSandbox


class TeslaSandboxProfilleyici:
    """
    Tesla Chromium Sandbox Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_sandbox(self) -> Dict[str, Any]:
        sandbox = TeslaChromiumSeccompSandbox()
        test_calls = [
            "read", "write", "mmap", "socket", "futex",
            "ptrace", "epoll_wait", "reboot", "clock_gettime", "bpf"
        ]

        gecikmeler_us: List[float] = []
        ciktilar = None

        for _ in range(self.iterations):
            t0 = time.perf_counter_ns()
            ciktilar = sandbox.evaluate_syscall_batch(test_calls)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))
        per_call_us = t_avg_us / len(test_calls)

        return {
            "syscall_check_ortalama_us": per_call_us,
            "batch_check_ortalama_us": t_avg_us,
            "batch_check_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_syscall_kontrolu": int(1e6 / max(per_call_us, 1e-4)),
            "permitted": ciktilar["permitted_count"],
            "blocked": ciktilar["blocked_count"],
            "blocked_list": ciktilar["blocked_syscalls"],
            "is_secure": ciktilar["sandbox_secure"],
            "gecikmeler": gecikmeler_us[:200]
        }
