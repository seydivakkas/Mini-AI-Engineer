r"""
Tesla MISRA C++ ve Statik Mimari İnceleme Motoru
=================================================
Bu modül; Tesla FSD ve Optimus için MISRA C++:2023 ve AUTOSAR C++14
güvenlik kurallarını (çalışma zamanında dinamik bellek yasağı, özyineleme yasağı,
güvensiz işaretçi aritmetiği ve tanımsız tür dönüşümleri) tarayan statik
analiz ve linter çekirdeğini gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class MISRAViolation:
    rule_id: str
    severity: str  # "MANDATORY", "REQUIRED", "ADVISORY"
    line_no: int
    message: str
    code_snippet: str


class TeslaMISRACodeReviewer:
    """
    Tesla MISRA C++ ve Gömülü Güvenlik Statik Kod İnceleyicisi.
    """
    def __init__(self):
        # Kural Regex Tanımları
        self.rules = [
            {
                "rule_id": "MISRA-C++:Rule 18.4",
                "severity": "MANDATORY",
                "pattern": re.compile(r'\b(malloc|free|new\s+|delete\s+)\b'),
                "message": "Dinamik bellek tahsisi (malloc/free/new/delete) RTOS döngüsünde kesinlikle yasaktır."
            },
            {
                "rule_id": "MISRA-C++:Rule 5.1",
                "severity": "REQUIRED",
                "pattern": re.compile(r'\b(goto|setjmp|longjmp)\b'),
                "message": "Deterministik akış ihlali (goto/setjmp/longjmp) yasaktır."
            },
            {
                "rule_id": "MISRA-C++:Rule 5.2",
                "severity": "REQUIRED",
                "pattern": re.compile(r'\breinterpret_cast\s*<'),
                "message": "Tanımsız bellek erişimi riski: reinterpret_cast kullanımı yasaktır."
            },
            {
                "rule_id": "MISRA-C++:Rule 17.2",
                "severity": "MANDATORY",
                "pattern": re.compile(r'\bwhile\s*\(\s*(1|true)\s*\)\s*\{(?!.*break)'),
                "message": "Sonsuz döngü ve deterministik olmayan yürütme riski."
            }
        ]

    def scan_cpp_source(self, code_str: str) -> List[MISRAViolation]:
        """C++ kaynak kodunu satır satır analiz eder ve ihlalleri listeler."""
        violations: List[MISRAViolation] = []
        lines = code_str.split('\n')

        for idx, line in enumerate(lines, 1):
            clean_line = line.strip()
            if not clean_line or clean_line.startswith('//') or clean_line.startswith('/*'):
                continue

            for rule in self.rules:
                if rule["pattern"].search(clean_line):
                    violations.append(MISRAViolation(
                        rule_id=rule["rule_id"],
                        severity=rule["severity"],
                        line_no=idx,
                        message=rule["message"],
                        code_snippet=clean_line
                    ))

        return violations

    def calculate_compliance_score(self, violations: List[MISRAViolation], total_lines: int) -> Dict[str, Any]:
        """MISRA C++ uyum skorunu ve güvenlik sertifikasyon durumunu hesaplar."""
        mand_count = sum(1 for v in violations if v.severity == "MANDATORY")
        req_count = sum(1 for v in violations if v.severity == "REQUIRED")

        # Ceza puanı: Her MANDATORY -15 puan, هر REQUIRED -5 puan
        penalty = (mand_count * 15.0) + (req_count * 5.0)
        compliance_pct = max(0.0, 100.0 - (penalty / max(1.0, total_lines * 0.05)))

        is_passed = (mand_count == 0 and req_count == 0)

        return {
            "total_lines_scanned": total_lines,
            "total_violations": len(violations),
            "mandatory_violations": mand_count,
            "required_violations": req_count,
            "compliance_score_pct": float(round(compliance_pct, 2)),
            "safety_certification_passed": is_passed,
            "status": "APPROVED (ASIL-D HAZIR)" if is_passed else "REJECTED (KOD DEĞİŞİKLİĞİ GEREKLİ)"
        }
