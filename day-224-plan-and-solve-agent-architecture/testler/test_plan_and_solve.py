"""
PyTest Birim Testleri - Day 224: Plan-and-Solve (PS+) Ajan Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.plan_and_solve_motoru import (
    SubTask,
    PlannerEngine,
    PlanAndSolveAgent,
)
from src.plan_profilleyici import PlanProfilleyici
from src.gorsellestirici import PlanAndSolveGorsellestirici


def test_subtask_initialization():
    """1. SubTask nesnesi alanları doğru başlatmalı ve özet üretmelidir."""
    task = SubTask(1, "Veri indir", "Indirici", {"url": "http://api"})
    assert task.gorev_id == 1
    assert task.durum == "BEKLIYOR"
    assert "Veri indir" in task.ozet()


def test_planner_engine_plan_creation():
    """2. PlannerEngine hedefi sıralı alt görev listesine dönüştürmelidir."""
    alt = [("Adım 1", "Arac1", {}), ("Adım 2", "Arac2", {})]
    plan = PlannerEngine.plan_olustur("Ana Hedef", alt)
    assert len(plan) == 2
    assert plan[0].gorev_id == 1
    assert plan[1].gorev_id == 2


def test_agent_tool_registration():
    """3. PlanAndSolveAgent harici araçları kaydetmelidir."""
    ajan = PlanAndSolveAgent()
    ajan.arac_kaydet("Kare", lambda x: int(x) ** 2)
    assert "Kare" in ajan._araclar


def test_sequential_execution_success():
    """4. PlanAndSolveAgent planı sırayla icra edip tüm görevleri tamamlamalıdır."""
    ajan = PlanAndSolveAgent()
    ajan.arac_kaydet("Adim1", lambda: "Veri1")
    ajan.arac_kaydet("Adim2", lambda: "Veri2")

    plan = [
        SubTask(1, "Görev 1", "Adim1"),
        SubTask(2, "Görev 2", "Adim2"),
    ]
    sonuc = ajan.plani_yurut("Hedef", plan)
    assert sonuc["tamamlandi_mi"] is True
    assert sonuc["tamamlanan_alt_gorev"] == 2


def test_memory_injection():
    """5. PlanAndSolveAgent önceki görev çıktılarını hafızadan enjekte edebilmelidir."""
    ajan = PlanAndSolveAgent()
    ajan.arac_kaydet("Topla", lambda a, b: a + b)
    ajan.arac_kaydet("IkiyeBol", lambda deger: float(deger) / 2.0)

    plan = [
        SubTask(1, "Toplama yap", "Topla", {"a": 20, "b": 30}),
        SubTask(2, "Sonucu böl", "IkiyeBol", {"deger": "$bellek.gorev_1_sonuc"}),
    ]
    sonuc = ajan.plani_yurut("Matematik", plan)
    assert sonuc["tamamlandi_mi"] is True
    assert sonuc["calisma_bellegi"]["gorev_2_sonuc"] == "25.0"


def test_error_handling_in_subtask():
    """6. PlanAndSolveAgent araç hatasını yakalayıp HATALI olarak işaretlemelidir."""
    ajan = PlanAndSolveAgent()

    def hatali_fonk():
        raise ValueError("Veritabanı bağlantısı koptu")

    ajan.arac_kaydet("HataUretici", hatali_fonk)

    plan = [SubTask(1, "Bozuk Görev", "HataUretici")]
    sonuc = ajan.plani_yurut("Hatalı Test", plan)
    assert sonuc["tamamlandi_mi"] is False
    assert plan[0].durum == "HATALI"


def test_profiler_plan_and_solve_metrics():
    """7. Profilleyici Plan-and-Solve mimarisinin tamamlama oranının %90 üstünde olduğunu göstermelidir."""
    prof = PlanProfilleyici.basarim_profili_cikar()
    skor = prof["karsilastirma"]["karmasik_gorev_tamamlama_orani"]["Plan_and_Solve_PS"]
    assert skor > 90.0


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. PlanAndSolveGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_plan_paneli.png")
    profil = PlanProfilleyici.basarim_profili_cikar()

    PlanAndSolveGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
