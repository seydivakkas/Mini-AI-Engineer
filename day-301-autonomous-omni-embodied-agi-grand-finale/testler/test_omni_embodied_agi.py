"""
PyTest Birim Testleri - Day 301 (BÜYÜK FİNAL): Uçtan Uca Bedenlenmiş Çok Modlu Otonom AGI Sistemi.
8/8 Kapsamlı Şampiyonluk Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.omni_embodied_agi_motoru import (
    OmniModalPerception,
    DeepReasoningCoT,
    HardwareKernelSubsystem,
    WorldModelEmbodiment,
    QuantumScientificSolver,
    OmniEmbodiedAGISystem,
)
from src.omni_embodied_agi_profilleyici import OmniEmbodiedAGIProfilleyici
from src.gorsellestirici import OmniEmbodiedAGIGorsellestirici


def test_omni_modal_perception_fusion():
    """1. Çok modlu algı motoru tüm duyusal girdileri 512 boyutlu latent vektörde birleştirmelidir."""
    fused = OmniModalPerception.fuse_sensory_inputs()
    assert fused["unified_latent_dim"] == 512
    assert len(fused["latent_vector"]) == 512
    assert fused["perception_confidence"] >= 0.99


def test_deep_reasoning_cot_deliberation():
    """2. GRPO derin akıl yürütme motoru geçerli <think> blokları ve yüksek güvenlik puanı üretmelidir."""
    reasoning = DeepReasoningCoT.deliberate("Test Görevi")
    assert "<think>" in reasoning["reasoning_trace"]
    assert "</think>" in reasoning["reasoning_trace"]
    assert reasoning["formal_proof_verified"] is True
    assert reasoning["safety_score"] >= 0.999


def test_hardware_kernel_subsystem_metrics():
    """3. Donanım hızlandırıcı alt sistemi 500+ MHz saat frekansı ve <10 ms gecikme sağlamalıdır."""
    hw = HardwareKernelSubsystem.execute_hardware_accelerator()
    assert hw["clock_frequency_mhz"] >= 500.0
    assert hw["latency_ms"] <= 10.0
    assert hw["energy_efficiency_tflops_per_watt"] >= 15.0


def test_world_model_embodiment_action_generation():
    """4. Dünya modeli robotik motoru 6-DoF eklem açıları üretmeli ve %95+ Sim-to-Real başarısı sunmalıdır."""
    latent = OmniModalPerception.fuse_sensory_inputs()["latent_vector"]
    act = WorldModelEmbodiment.generate_robot_action(latent)
    assert len(act["joint_angles_rad"]) == 6
    assert act["sim_to_real_success_pct"] >= 95.0
    assert act["tactile_slip_detected"] is False


def test_quantum_scientific_solver_accuracy():
    """5. Kuantum bilimsel çözücü kimyasal hassasiyet sınırını (< 1.6 mHa) sağlamalıdır."""
    q = QuantumScientificSolver.solve_molecular_energy()
    assert q["chemical_accuracy_met"] is True
    assert q["energy_error_hartree"] < 0.0016


def test_omni_embodied_agi_full_autonomous_cycle():
    """6. Birleşik AGI orkestratörü tüm 15 fazın alt modüllerini hatasız koşturmalıdır."""
    cycle = OmniEmbodiedAGISystem.run_full_autonomous_cycle()
    assert "perception" in cycle
    assert "reasoning" in cycle
    assert "hardware" in cycle
    assert "embodiment" in cycle
    assert "quantum" in cycle
    assert "301 GÜNLÜK" in cycle["grand_finale_status"]


def test_profiler_mmlu_and_latency_speedup():
    """7. Profilleyici 95+ MMLU skoru ve 15x üzeri uçtan uca hızlanma raporlamalıdır."""
    profil = OmniEmbodiedAGIProfilleyici.basarim_profili_cikar()
    mmlu = profil["karsilastirma"]["cok_modlu_mmlu_skoru"]["3. Omni-Embodied AGI (301)"]
    assert mmlu >= 95.0
    assert profil["hizlanma_carpani"] >= 15.0


def test_gorsellestirici_grand_finale_dashboard_creation(tmp_path):
    """8. OmniEmbodiedAGIGorsellestirici 6 panelli büyük final teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_grand_finale_paneli.png")
    profil = OmniEmbodiedAGIProfilleyici.basarim_profili_cikar()

    OmniEmbodiedAGIGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
