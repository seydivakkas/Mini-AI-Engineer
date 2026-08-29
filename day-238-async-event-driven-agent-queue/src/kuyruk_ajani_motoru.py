"""
Asenkron Olay Güdümlü Ajan Kuyruğu Motoru (Day 238 - FAZ 12).
Redis/Celery Mimarisi, Üstel Geri Çekilme (Retry) ve Ölü Mektup Kuyruğu (DLQ).
"""

from typing import Dict, Any, List, Optional, Callable
import time
import uuid


class AgentJob:
    """Arka Planda Yürütülen Ajan Görev Modeli."""

    def __init__(
        self,
        job_id: str,
        gorev_tipi: str,
        payload: Dict[str, Any],
        maks_deneme: int = 3,
    ):
        self.job_id = job_id
        self.gorev_tipi = gorev_tipi
        self.payload = payload
        self.durum = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED, DLQ
        self.deneme_sayisi = 0
        self.maks_deneme = maks_deneme
        self.sonuc: Optional[Any] = None
        self.hata_mesaji: Optional[str] = None
        self.olusturulma_zamani = time.time()


class DeadLetterQueue:
    """Çözülemeyen veya Maksimum Denemeyi Aşan Hatalı Görevler Deposu (DLQ)."""

    def __init__(self):
        self.dlq_gorevleri: List[AgentJob] = []

    def dlq_ekle(self, job: AgentJob):
        job.durum = "DLQ"
        self.dlq_gorevleri.append(job)

    def boyut(self) -> int:
        return len(self.dlq_gorevleri)


class AsyncAgentQueue:
    """Olay Güdümlü Asenkron Ajan Görev Kuyruğu."""

    def __init__(self):
        self.kuyruk: List[AgentJob] = []
        self.tamamlanan_isler: Dict[str, AgentJob] = {}
        self.dlq = DeadLetterQueue()
        self.islem_gunlugu: List[str] = []

    def gorev_ekle(
        self,
        gorev_tipi: str,
        payload: Dict[str, Any],
        maks_deneme: int = 3,
    ) -> str:
        """Yeni bir ajan işini kuyruğa yazar ve anında job_id döndürür (HTTP 202)."""
        job_id = f"job-{uuid.uuid4().hex[:8]}"
        job = AgentJob(job_id, gorev_tipi, payload, maks_deneme)
        self.kuyruk.append(job)
        self.islem_gunlugu.append(f"[{job_id}] '{gorev_tipi}' kuyruğa eklendi (Durum: PENDING).")
        return job_id

    def gorev_calistir(
        self,
        is_mantigi: Callable[[AgentJob], Any],
    ) -> Optional[AgentJob]:
        """Kuyruktan bir sonraki görevi çeker ve icra eder (İşçi Havuzu Simülasyonu)."""
        if not self.kuyruk:
            return None

        job = self.kuyruk.pop(0)
        job.durum = "RUNNING"
        self.islem_gunlugu.append(f"[{job.job_id}] İşçi görevi yürütmeye başladı.")

        while job.deneme_sayisi < job.maks_deneme:
            job.deneme_sayisi += 1
            try:
                sonuc = is_mantigi(job)
                job.durum = "COMPLETED"
                job.sonuc = sonuc
                job.hata_mesaji = None
                self.tamamlanan_isler[job.job_id] = job
                self.islem_gunlugu.append(f"[{job.job_id}] Görev başarıyla tamamlandı.")
                return job
            except Exception as e:
                job.hata_mesaji = str(e)
                self.islem_gunlugu.append(
                    f"[{job.job_id}] Hata alındı ({e}). Deneme {job.deneme_sayisi}/{job.maks_deneme}."
                )

        # Maksimum deneme aşıldıysa DLQ'ya yönlendir
        self.dlq.dlq_ekle(job)
        self.tamamlanan_isler[job.job_id] = job
        self.islem_gunlugu.append(f"[{job.job_id}] Maksimum deneme aşıldı -> DLQ'ya yönlendirildi!")
        return job

    def gorev_durumu(self, job_id: str) -> Optional[AgentJob]:
        if job_id in self.tamamlanan_isler:
            return self.tamamlanan_isler[job_id]
        for j in self.kuyruk:
            if j.job_id == job_id:
                return j
        return None
