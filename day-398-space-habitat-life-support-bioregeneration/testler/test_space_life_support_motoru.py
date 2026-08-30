"""
Day 398: Unit Tests for Autonomous Deep-Space Habitat Life Support & Bioregeneration AI
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved
"""

import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from space_life_support_motoru import (
    HabitatChamberState,
    SabatierElectrolysisReactor,
    MicroalgaePhotobioreactor,
    ECLSSNonlinearMPC,
    SpaceHabitatBenchmark
)


def test_sabatier_electrolysis_stoichiometry():
    """Sabatier reaktörünün kütle korunumu ve O2 geri kazanımını test eder."""
    reactor = SabatierElectrolysisReactor(efficiency=0.985)
    o2_rec, w_rec, ch4 = reactor.process_day(co2_input_kg=4.0, available_water_l=1000.0)

    assert o2_rec > 2.5
    assert w_rec > 3.0
    assert ch4 > 1.0


def test_microalgae_photobioreactor_growth():
    """Mikroalg fotobiyoreaktörünün O2 üretimi ve biyokütle artışını test eder."""
    algae = MicroalgaePhotobioreactor(initial_biomass_kg=50.0)
    o2_prod, new_bio = algae.grow_day(co2_consumed_kg=2.0, photon_flux_par=400.0)

    assert o2_prod > 1.5
    assert new_bio >= 50.0


def test_eclss_mpc_control_action():
    """MPC kontrolcüsünün düşük O2 anında elektroliz oranını artırdığını test eder."""
    mpc = ECLSSNonlinearMPC()
    elec_rate_low_o2, _ = mpc.compute_control_action(current_po2=20.0, current_pco2=0.35)
    elec_rate_high_o2, _ = mpc.compute_control_action(current_po2=21.8, current_pco2=0.35)

    assert elec_rate_low_o2 > elec_rate_high_o2


def test_tam_space_habitat_benchmark():
    """Tam derin uzay yaşam destek benchmarkını test eder."""
    bench = SpaceHabitatBenchmark(mission_days=50, crew_count=4)
    res = bench.kos()

    assert res["mission_days"] == 50
    assert res["closure_loop_pct"] >= 98.0
    assert res["hypoxia_incidents"] == 0
    assert 20.5 <= res["avg_po2_kpa"] <= 21.5
