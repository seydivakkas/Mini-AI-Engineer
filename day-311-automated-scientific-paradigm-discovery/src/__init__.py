"""
Day 311: Automated Scientific Theory & Paradigm Discovery Engine.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from .bilimsel_kesif_motoru import (
    ScientificDiscoveryConfig,
    ScientificDiscoveryResult,
    CandidateLibrary,
    SINDyEquationDiscoverer,
    AutomatedScientificDiscoveryEngine
)
from .bilimsel_kesif_profilleyici import ScientificDiscoveryProfiler
from .gorsellestirici import ScientificDiscoveryGorsellestirici

__all__ = [
    "ScientificDiscoveryConfig",
    "ScientificDiscoveryResult",
    "CandidateLibrary",
    "SINDyEquationDiscoverer",
    "AutomatedScientificDiscoveryEngine",
    "ScientificDiscoveryProfiler",
    "ScientificDiscoveryGorsellestirici"
]
