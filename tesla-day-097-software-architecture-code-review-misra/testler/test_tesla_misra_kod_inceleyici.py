"""
Tesla MISRA C++ Kod İnceleyici Birim Testleri (PyTest)
======================================================
Bu test paketi; dinamik bellek, tanımsız tür dönüşümleri ve
ISO 26262 ASIL-D güvenlik kurallarının tespitini test eder.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import sys
import os

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
ana_dizin = os.path.dirname(su_an_dizin)
if ana_dizin not in sys.path:
    sys.path.insert(0, ana_dizin)

from src.tesla_misra_kod_inceleyici import TeslaMISRACodeReviewer


def test_dinamik_bellek_ve_reinterpret_cast_tespiti():
    """malloc ve reinterpret_cast ihlallerinin başarıyla yakalandığı test edilir."""
    reviewer = TeslaMISRACodeReviewer()

    bad_code = """
    void bad_function() {
        int* buffer = (int*)malloc(1024);
        char* raw_ptr = reinterpret_cast<char*>(buffer);
        free(buffer);
    }
    """

    violations = reviewer.scan_cpp_source(bad_code)
    assert len(violations) >= 3

    rule_ids = [v.rule_id for v in violations]
    assert "MISRA-C++:Rule 18.4" in rule_ids
    assert "MISRA-C++:Rule 5.2" in rule_ids


def test_temiz_kod_ve_asil_d_uyumu():
    """MISRA kurallarına tam uyan C++ kodunun %100 skor aldığı test edilir."""
    reviewer = TeslaMISRACodeReviewer()

    clean_code = """
    #include <array>
    #include <cstdint>

    void safe_torque_clamp(std::array<float, 6>& torques) {
        for (auto& t : torques) {
            if (t > 150.0f) {
                t = 150.0f;
            } else if (t < -150.0f) {
                t = -150.0f;
            }
        }
    }
    """

    violations = reviewer.scan_cpp_source(clean_code)
    assert len(violations) == 0

    score_res = reviewer.calculate_compliance_score(violations, total_lines=20)
    assert score_res["compliance_score_pct"] == 100.0
    assert score_res["safety_certification_passed"] is True
    assert "APPROVED" in score_res["status"]
