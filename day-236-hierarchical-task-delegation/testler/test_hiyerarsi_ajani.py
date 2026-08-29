"""
PyTest Birim Testleri - Day 236: Hiyerarşik Görev Delegasyonu Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.hiyerarsi_ajani_motoru import (
    SubTask,
    WorkerAgent,
    ManagerAgent,
)
from src.hiyerarsi_profilleyici import HiyerarsiProfilleyici
from src.gorsellestirici import HiyerarsiGorsellestirici


def test_subtask_initialization():
    """1. SubTask alanları doğru başlatmalıdır."""
    task = SubTask(1, "Database", "Tablo oluştur")
    assert task.task_id == 1
    assert task.alan == "database"
    assert task.durum == "BEKLIYOR"


def test_worker_agent_execution_db():
    """2. Database işçisi veritabanı görevini başarıyla tamamlamalıdır."""
    worker = WorkerAgent("database", "DB Uzmanı")
    task = SubTask(1, "database", "Users tablosu")
    out = worker.gorev_yurut(task)
    assert "users" in out
    assert task.durum == "TAMAMLANDI"


def test_worker_agent_execution_backend():
    """3. Backend işçisi API görevini başarıyla tamamlamalıdır."""
    worker = WorkerAgent("backend", "FastAPI Uzmanı")
    task = SubTask(2, "backend", "Auth endpoint")
    out = worker.gorev_yurut(task)
    assert "FastAPI" in out
    assert task.durum == "TAMAMLANDI"


def test_worker_agent_execution_security():
    """4. Security işçisi güvenlik görevini başarıyla tamamlamalıdır."""
    worker = WorkerAgent("security", "Güvenlik Uzmanı")
    task = SubTask(3, "security", "Rate limit")
    out = worker.gorev_yurut(task)
    assert "Bcrypt" in out
    assert task.durum == "TAMAMLANDI"


def test_manager_agent_decomposition():
    """5. ManagerAgent kök hedefi 3 alt göreve ayrıştırmalıdır."""
    manager = ManagerAgent()
    tasks = manager.gorevi_ayristir("Mikroservis Kur")
    assert len(tasks) == 3


def test_manager_delegation_execution():
    """6. ManagerAgent tüm işçilere görevleri dağıtıp sentez oluşturmalıdır."""
    manager = ManagerAgent()
    manager.isci_kaydet(WorkerAgent("database", "DB Uzmanı"))
    manager.isci_kaydet(WorkerAgent("backend", "Backend Uzmanı"))
    manager.isci_kaydet(WorkerAgent("security", "Güvenlik Uzmanı"))

    sonuc = manager.gorevleri_delege_et_ve_birlestir("Servis Kur")
    assert len(sonuc["tamamlanan_ciktilar"]) == 3
    assert "başarıyla tamamlandı" in sonuc["sentez"]


def test_profiler_metrics_comparison():
    """7. Profilleyici Hiyerarşik Ajan başarısının %90 üzerinde olduğunu doğrulamalıdır."""
    prof = HiyerarsiProfilleyici.basarim_profili_cikar()
    skor = prof["karsilastirma"]["karmasik_gorev_basarisi"]["Hiyerarsik_Yonetici"]
    assert skor > 90.0


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. HiyerarsiGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_hiyerarsi_paneli.png")
    profil = HiyerarsiProfilleyici.basarim_profili_cikar()

    HiyerarsiGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
