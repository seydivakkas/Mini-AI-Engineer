"""
PyTest Birim Testleri - Day 210: Self-Play RL ve Sentetik Veri Döngüsü.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.self_play_motoru import (
    SyntheticProblemGenerator,
    ReasoningSolver,
    SelfPlayReferee,
    CurriculumScheduler,
    SelfPlayRLTrainer,
)
from src.self_play_profilleyici import SelfPlayProfilleyici
from src.gorsellestirici import SelfPlayGorsellestirici


def test_problem_generator_levels():
    """1. Problem üretici geçerli soru sözlüğü üretmelidir."""
    p1 = SyntheticProblemGenerator.problem_uret(zorluk=1.0)
    p5 = SyntheticProblemGenerator.problem_uret(zorluk=5.0)
    p9 = SyntheticProblemGenerator.problem_uret(zorluk=9.0)

    assert "soru" in p1 and "dogru_cevap" in p1
    assert "soru" in p5 and "dogru_cevap" in p5
    assert "soru" in p9 and "dogru_cevap" in p9


def test_solver_execution():
    """2. ReasoningSolver düşünce zinciri ve yanıt üretmelidir."""
    solver = ReasoningSolver(yetenek_theta=3.0)
    p = SyntheticProblemGenerator.problem_uret(zorluk=2.0)
    cozum = solver.cozumu_yurut(p)

    assert "<think>" in cozum["dusunce_yolu"]
    assert "uretilen_cevap" in cozum


def test_referee_reward_correct():
    """3. Hakem doğru çözüme 1.0 puan vermelidir."""
    p = {"soru": "x=2", "dogru_cevap": "2"}
    cozum = {"uretilen_cevap": "2", "p_basari_tahmini": 0.5}
    odul = SelfPlayReferee.odul_hesapla(p, cozum)

    assert odul["r_solver"] == 1.0
    assert odul["dogru_mu"] is True


def test_referee_generator_reward_bounds():
    """4. Üretici ödülü [0.0, 1.0] aralığında olmalıdır."""
    p = {"soru": "x=2", "dogru_cevap": "2"}
    cozum = {"uretilen_cevap": "3", "p_basari_tahmini": 0.5}
    odul = SelfPlayReferee.odul_hesapla(p, cozum)

    assert 0.0 <= odul["r_generator"] <= 1.0


def test_curriculum_scheduler_increase():
    """5. Sürekli başarılı olunduğunda zorluk seviyesi artmalıdır."""
    sched = CurriculumScheduler(baslangic_zorluk=2.0)
    for _ in range(8):
        sched.sonucu_kaydet_ve_zorlugu_guncelle(True)

    assert sched.zorluk > 2.0


def test_curriculum_scheduler_decrease():
    """6. Sürekli başarısız olunduğunda zorluk seviyesi azalmalıdır."""
    sched = CurriculumScheduler(baslangic_zorluk=5.0)
    for _ in range(8):
        sched.sonucu_kaydet_ve_zorlugu_guncelle(False)

    assert sched.zorluk < 5.0


def test_self_play_simulation_growth():
    """7. SelfPlay simülasyonu metrikleri eksiksiz kaydetmelidir."""
    profil = SelfPlayProfilleyici.simulasyon_yurut(toplam_tur=15)
    assert len(profil["turlar"]) == 15
    assert "zorluk_egrisi" in profil
    assert "yetenek_egrisi" in profil


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. SelfPlayGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_self_play_paneli.png")
    profil = SelfPlayProfilleyici.simulasyon_yurut(toplam_tur=20)

    SelfPlayGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
