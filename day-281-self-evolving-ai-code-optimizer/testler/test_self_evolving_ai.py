"""
PyTest Birim Testleri - Day 281 (FAZ 15): Self-Evolving AI Kod ve Çekirdek Optimize Edici.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.self_evolving_motoru import SelfEvolvingAIEngine, KernelGenome
from src.self_evolving_profilleyici import SelfEvolvingProfilleyici
from src.gorsellestirici import SelfEvolvingGorsellestirici


def test_kernel_genome_initialization_and_to_dict():
    """1. KernelGenome doğru başlangıç değerlerini ve sözlük temsilini üretmelidir."""
    genome = KernelGenome(block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=3, unroll_factor=2)
    d = genome.to_dict()
    assert d["BLOCK_M"] == 64
    assert d["BLOCK_N"] == 64
    assert d["num_warps"] == 4
    assert d["num_stages"] == 3


def test_kernel_genome_mutation():
    """2. Genom mutasyonu geçerli donanım aralıklarında yeni konfigürasyon üretmelidir."""
    genome = KernelGenome()
    mutant = genome.mutate()
    assert mutant.block_m in [16, 32, 64, 128, 256]
    assert mutant.num_warps in [2, 4, 8, 16]
    assert mutant.num_stages in [2, 3, 4, 5]


def test_ast_parse_and_validation():
    """3. Python AST ayrıştırıcı fonksiyon düğümlerini ve geçerliliği doğru tespit etmelidir."""
    code = "def custom_triton_kernel(x, y): return x + y"
    res = SelfEvolvingAIEngine.parse_and_validate_ast(code)
    assert res["is_valid"] is True
    assert "custom_triton_kernel" in res["function_names"]
    assert res["total_ast_nodes"] > 0


def test_evaluate_kernel_fitness_numerical_check():
    """4. Fitness değerlendirmesi pozitif TFLOPS ve doğrulanmış hata (<1e-4) döndürmelidir."""
    genome = KernelGenome(block_m=128, block_n=128, num_warps=8, num_stages=4)
    tflops, err, is_valid = SelfEvolvingAIEngine.evaluate_kernel_fitness(genome)
    assert tflops > 100.0
    assert err < 1e-4
    assert is_valid is True


def test_run_evolutionary_optimization_progress():
    """5. Otonom genetik evrim başlangıç durumuna göre en az 1.8x hızlanma sağlamalıdır."""
    res = SelfEvolvingAIEngine.run_evolutionary_optimization(generations=5, population_size=8)
    assert res["speedup_ratio"] > 1.8
    assert res["final_tflops"] > res["initial_tflops"]


def test_evolution_trajectory_length():
    """6. Evrim yörüngesi başlangıç dahil tüm nesilleri (0-5) içermelidir."""
    res = SelfEvolvingAIEngine.run_evolutionary_optimization(generations=5, population_size=6)
    assert len(res["trajectory"]) == 6
    assert res["trajectory"][0]["generation"] == 0
    assert res["trajectory"][-1]["generation"] == 5


def test_profiler_metrics_and_hot_patch_latency():
    """7. Profilleyici sıcak-yenileme gecikmesini (<1ms) ve doğrulama geçerliliğini (%100) doğrulamalıdır."""
    profil = SelfEvolvingProfilleyici.basarim_profili_cikar()
    kars = profil["karsilastirma"]
    assert kars["dogrulama_gecerlilik_orani_yuzde"]["Gen_5_Self_Evolved"] == 100.0
    assert kars["hot_patching_gecikmesi_ms"]["Gen_5_Self_Evolved"] < 1.0
    assert profil["hizlanma_orani"] > 1.8


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. SelfEvolvingGorsellestirici 6 panelli teşhis panosunu başarıyla kaydetmelidir."""
    cikti = str(tmp_path / "test_self_evolving_paneli.png")
    profil = SelfEvolvingProfilleyici.basarim_profili_cikar()

    SelfEvolvingGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
