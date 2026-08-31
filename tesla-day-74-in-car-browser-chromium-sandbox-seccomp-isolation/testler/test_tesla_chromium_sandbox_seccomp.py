"""
Tesla Chromium Sandbox Birim Testleri (PyTest)
==============================================
Bu test paketi; Seccomp-BPF sistem çağrısı filtre kurallarını,
tehlikeli çağrıların engellenmesini ve kum havuzu bütünlüğünü test eder.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import numpy as np
import sys
import os

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
ana_dizin = os.path.dirname(su_an_dizin)
if ana_dizin not in sys.path:
    sys.path.insert(0, ana_dizin)

from src.tesla_chromium_sandbox_seccomp import TeslaChromiumSeccompSandbox


def test_guvenli_cagrilarin_onaylanmasi():
    """Temel okuma/yazma ve bellek çağrılarına izin verildiği test edilir."""
    sandbox = TeslaChromiumSeccompSandbox()

    assert sandbox.is_syscall_permitted("read") is True
    assert sandbox.is_syscall_permitted("write") is True
    assert sandbox.is_syscall_permitted("mmap") is True
    assert sandbox.is_syscall_permitted("futex") is True


def test_tehlikeli_cagrilarin_engellenmesi():
    """Socket, ptrace ve reboot gibi kritik çağrıların kesin olarak engellendiği test edilir."""
    sandbox = TeslaChromiumSeccompSandbox()

    assert sandbox.is_syscall_permitted("socket") is False
    assert sandbox.is_syscall_permitted("ptrace") is False
    assert sandbox.is_syscall_permitted("reboot") is False
    assert sandbox.is_syscall_permitted("kexec_load") is False


def test_toplu_syscall_guvenlik_denetimi():
    """Toplu çağrı paketinin doğru ayrıştırıldığı ve güvenliğin sağlandığı test edilir."""
    sandbox = TeslaChromiumSeccompSandbox()
    batch = ["read", "write", "socket", "ptrace", "mmap"]
    res = sandbox.evaluate_syscall_batch(batch)

    assert res["total_requests"] == 5
    assert res["permitted_count"] == 3
    assert res["blocked_count"] == 2
    assert "socket" in res["blocked_syscalls"]
    assert "ptrace" in res["blocked_syscalls"]
    assert res["sandbox_secure"] is True
