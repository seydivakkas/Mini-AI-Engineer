"""
PyTest Birim Testleri - Day 237: Ajan Öz-Yansıtma ve Öz-Değerlendirme Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.refleksiyon_ajani_motoru import (
    EvaluationScore,
    ReflectionCritic,
    SelfRefiningAgent,
)
from src.refleksiyon_profilleyici import RefleksiyonProfilleyici
from src.gorsellestirici import RefleksiyonGorsellestirici


def test_evaluation_score_fields():
    """1. EvaluationScore toplam skoru ve onay durumunu doğru hesaplamalıdır."""
    rep = EvaluationScore(40.0, 40.0, 20.0, "Kusursuz", 90.0)
    assert rep.toplam_skor == 100.0
    assert rep.onaylandi_mi is True


def test_critic_flawed_code_evaluation():
    """2. ReflectionCritic düz metin şifre kontrolüne düşük güvenlik skoru vermelidir."""
    kod = "def check(p, h): return p == h"
    score = ReflectionCritic.degerlendir(kod)
    assert score.guvenlik <= 10.0
    assert score.toplam_skor < 90.0
    assert score.onaylandi_mi is False


def test_critic_intermediate_code_evaluation():
    """3. ReflectionCritic sha256 kullanımını tespit edip orta düzey skor vermelidir."""
    kod = "import hashlib\ndef check(p: str, h: str) -> bool: return hashlib.sha256(p).hexdigest() == h"
    score = ReflectionCritic.degerlendir(kod)
    assert score.toplam_skor >= 60.0


def test_critic_perfect_code_evaluation():
    """4. ReflectionCritic bcrypt ve try/except içeren güvenli koda tam puan vermelidir."""
    kod = "import bcrypt\ndef check(p: str, h: str) -> bool:\n    try:\n        return bcrypt.checkpw(p.encode(), h.encode())\n    except Exception:\n        return False"
    score = ReflectionCritic.degerlendir(kod)
    assert score.toplam_skor == 100.0
    assert score.onaylandi_mi is True


def test_self_refining_agent_convergence():
    """5. SelfRefiningAgent onay eşiğine ulaşan taslakta döngüyü başarıyla sonlandırmalıdır."""
    agent = SelfRefiningAgent(esik_puani=90.0, maks_iterasyon=3)
    t1 = "def f(a): return a"
    t2 = "import bcrypt\ndef f(a: str, b: str) -> bool:\n    try:\n        return bcrypt.checkpw(a.encode(), b.encode())\n    except Exception:\n        return False"
    sonuc = agent.iyilestir_ve_tamamla([t1, t2])
    assert sonuc["onaylandi"] is True
    assert sonuc["toplam_iterasyon"] == 2


def test_self_refining_agent_max_iterations():
    """6. SelfRefiningAgent eşik geçilemezse maks_iterasyonda durmalıdır."""
    agent = SelfRefiningAgent(esik_puani=99.0, maks_iterasyon=2)
    t1 = "def f(a): return a"
    t2 = "def f2(a): return a"
    sonuc = agent.iyilestir_ve_tamamla([t1, t2])
    assert sonuc["toplam_iterasyon"] == 2
    assert sonuc["onaylandi"] is False


def test_profiler_reflection_metrics():
    """7. Profilleyici Öz-Yansıtma Ajanı başarısının %90 üzerinde olduğunu doğrulamalıdır."""
    prof = RefleksiyonProfilleyici.basarim_profili_cikar()
    skor = prof["karsilastirma"]["guvenlik_ve_dogruluk_skoru"]["Yinelemeli_Oz_Yansitma"]
    assert skor > 90.0


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. RefleksiyonGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_refleksiyon_paneli.png")
    profil = RefleksiyonProfilleyici.basarim_profili_cikar()

    RefleksiyonGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
