"""
PyTest Birim Testleri - Day 238: Asenkron Olay Güdümlü Ajan Kuyruğu Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.kuyruk_ajani_motoru import (
    AgentJob,
    DeadLetterQueue,
    AsyncAgentQueue,
)
from src.kuyruk_profilleyici import KuyrukProfilleyici
from src.gorsellestirici import KuyrukGorsellestirici


def test_agent_job_initialization():
    """1. AgentJob alanları doğru başlatmalıdır."""
    job = AgentJob("j-1", "test_task", {"key": "val"}, 3)
    assert job.job_id == "j-1"
    assert job.durum == "PENDING"
    assert job.deneme_sayisi == 0


def test_dead_letter_queue_push():
    """2. DeadLetterQueue hatalı görevleri DLQ durumuna getirip saklamalıdır."""
    dlq = DeadLetterQueue()
    job = AgentJob("j-err", "faulty", {})
    dlq.dlq_ekle(job)
    assert dlq.boyut() == 1
    assert job.durum == "DLQ"


def test_async_queue_task_enqueuing():
    """3. AsyncAgentQueue yeni görevi kuyruğa alıp job_id döndürmelidir."""
    q = AsyncAgentQueue()
    jid = q.gorev_ekle("scrape", {"url": "https://example.com"})
    assert jid.startswith("job-")
    assert len(q.kuyruk) == 1


def test_async_queue_task_success_execution():
    """4. AsyncAgentQueue başarılı görevi COMPLETED durumuna getirmelidir."""
    q = AsyncAgentQueue()
    jid = q.gorev_ekle("calc", {"a": 2, "b": 3})
    job = q.gorev_calistir(lambda j: j.payload["a"] + j.payload["b"])
    assert job.durum == "COMPLETED"
    assert job.sonuc == 5


def test_async_queue_task_retry_and_dlq():
    """5. AsyncAgentQueue hata durumunda yeniden deneyip DLQ'ya yönlendirmelidir."""
    q = AsyncAgentQueue()
    q.gorev_ekle("failing_task", {}, maks_deneme=2)

    def error_func(j):
        raise ValueError("Simulated error")

    job = q.gorev_calistir(error_func)
    assert job.durum == "DLQ"
    assert job.deneme_sayisi == 2
    assert q.dlq.boyut() == 1


def test_async_queue_status_lookup():
    """6. AsyncAgentQueue job_id ile görev durumunu getirebilmelidir."""
    q = AsyncAgentQueue()
    jid = q.gorev_ekle("lookup_test", {})
    job = q.gorev_durumu(jid)
    assert job is not None
    assert job.job_id == jid


def test_profiler_queue_metrics():
    """7. Profilleyici Olay Güdümlü Kuyrukta görev kaybının %0 olduğunu doğrulamalıdır."""
    prof = KuyrukProfilleyici.basarim_profili_cikar()
    kayip = prof["karsilastirma"]["gorev_kaybi_orani"]["Olay_Gudumlu_DLQ"]
    assert kayip == 0.0


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. KuyrukGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_kuyruk_paneli.png")
    profil = KuyrukProfilleyici.basarim_profili_cikar()

    KuyrukGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
