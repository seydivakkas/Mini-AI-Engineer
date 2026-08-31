"""
Tesla MISRA Profilleyici Modülü
===============================
Bu modül; MISRA C++ statik analiz motorunun satır işleme ve güvenlik
kuralı denetim hızını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_misra_kod_inceleyici import TeslaMISRACodeReviewer


class TeslaMISRAProfilleyici:
    """
    Tesla MISRA C++ Statik Analiz Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 50):
        self.iterations = iterations

    def benchmark_misra_scanner(self) -> Dict[str, Any]:
        reviewer = TeslaMISRACodeReviewer()

        # 500 satırlık sentetik C++ kodu
        sample_clean_cpp = """
        #include <cstdint>
        #include <array>

        class TeslaActuatorController {
        private:
            std::array<float, 6> m_joint_torques;
            float m_max_torque_limit{150.0f};

        public:
            TeslaActuatorController() : m_joint_torques{} {}

            void update_torque(const std::array<float, 6>& cmd) {
                for (size_t i = 0; i < 6; ++i) {
                    if (cmd[i] > m_max_torque_limit) {
                        m_joint_torques[i] = m_max_torque_limit;
                    } else if (cmd[i] < -m_max_torque_limit) {
                        m_joint_torques[i] = -m_max_torque_limit;
                    } else {
                        m_joint_torques[i] = cmd[i];
                    }
                }
            }
        };
        """ * 20

        gecikmeler_us: List[float] = []

        for _ in range(self.iterations):
            r_inst = TeslaMISRACodeReviewer()
            t0 = time.perf_counter_ns()
            _ = r_inst.scan_cpp_source(sample_clean_cpp)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        violations = reviewer.scan_cpp_source(sample_clean_cpp)
        total_lines = len(sample_clean_cpp.split('\n'))
        score_res = reviewer.calculate_compliance_score(violations, total_lines)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))
        per_line_us = t_avg_us / max(1, total_lines)

        return {
            "total_lines_scanned": total_lines,
            "violations_found": len(violations),
            "compliance_score_pct": score_res["compliance_score_pct"],
            "status": score_res["status"],
            "step_ortalama_us": t_avg_us,
            "per_line_us": per_line_us,
            "step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_satir_tarama": int(1e6 / max(per_line_us, 1e-4)),
            "gecikmeler": gecikmeler_us[:200]
        }
