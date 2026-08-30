"""
Day 383: Unit Tests for Autonomous Drug Discovery & Molecular Dynamics Simulation
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved
"""

import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from molecular_dynamics_engine import (
    Atom3D,
    AMBERForceField,
    VelocityVerletIntegrator,
    MMPBSAEstimator,
    MolecularDynamicsBenchmark
)


def test_amber_force_field_computation():
    """AMBER kuvvet alanının Lennard-Jones ve Coulomb kuvvetlerini doğru hesapladığını test eder."""
    ff = AMBERForceField(cutoff_angstrom=10.0)
    a1 = Atom3D(atom_id=0, element="C", residue="ALA", is_ligand=False, pos=np.array([0.0, 0.0, 0.0]), charge_e=0.2)
    a2 = Atom3D(atom_id=1, element="O", residue="ALA", is_ligand=False, pos=np.array([3.0, 0.0, 0.0]), charge_e=-0.2)

    vdw, elec, pot = ff.compute_forces_and_potential([a1, a2])
    assert pot != 0.0
    assert np.allclose(a1.force, -a2.force), "Newton'un 3. Hareket Yasası (Etki-Tepki) korunmalıdır."


def test_velocity_verlet_integrator_temperature():
    """Velocity-Verlet ve Langevin termostatının sıcaklığı kararlı tuttuğunu test eder."""
    integrator = VelocityVerletIntegrator(dt_fs=2.0, target_temp_k=300.0)
    ff = AMBERForceField()
    atoms = [
        Atom3D(atom_id=0, element="C", residue="ALA", is_ligand=False, pos=np.array([0.0, 0.0, 0.0])),
        Atom3D(atom_id=1, element="N", residue="ALA", is_ligand=False, pos=np.array([2.5, 0.0, 0.0]))
    ]

    kin_e, pot_e, temp = integrator.step(atoms, ff)
    assert kin_e >= 0.0
    assert temp >= 0.0


def test_mmpbsa_binding_free_energy():
    """MM-PBSA bağlanma serbest enerjisinin termodinamik çevrimle doğru hesaplandığını test eder."""
    mmpbsa = MMPBSAEstimator()
    res = mmpbsa.estimate_binding_free_energy(delta_vdw=-30.0, delta_elec=-20.0)

    assert "delta_g_bind_kcal_mol" in res
    assert res["delta_g_bind_kcal_mol"] < 0.0, "Spontan bağlanma için Delta G negatif olmalıdır."
    assert res["binding_affinity_nm"] > 0.0


def test_tam_molecular_dynamics_benchmark():
    """Tam Protein-Ligand MD simülasyonu ve ADMET profilini test eder."""
    bench = MolecularDynamicsBenchmark()
    res = bench.kos(num_steps=30)

    assert res["num_steps"] == 30
    assert res["final_rmsd_angstrom"] < 2.5, "Protein omurgası kararlı kalmalıdır."
    assert res["admet_profile"]["lipinski_rule_compliant"] is True
