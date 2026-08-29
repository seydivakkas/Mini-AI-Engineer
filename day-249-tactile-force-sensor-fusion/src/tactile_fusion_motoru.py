"""
Dokunsal (GelSight) ve 6-Eksenli Kuvvet Sensörü Füzyonu Motoru (Day 249).
Basınç Deformasyonu, Kayma (Slip) Tespiti ve 1000Hz Adaptif Tutuş Denetleyicisi.
"""

from typing import Dict, Any, List, Tuple
import numpy as np


class GelSightTactileSensor:
    """Jel Yüzey Deformasyonundan Basınç ve Temas Alanı Üreten Optik Dokunsal Sensör."""

    def __init__(self, res: int = 32):
        self.res = res

    def get_contact_patch(self, normal_kuvvet: float, merkez: Tuple[int, int] = (16, 16)) -> Dict[str, Any]:
        """Normal kuvvete bağlı olarak jel üzerindeki 2D basınç dağılımını hesaplar."""
        basinc_haritasi = np.zeros((self.res, self.res), dtype=np.float32)
        cx, cy = merkez
        # Normal kuvvet arttıkça temas yarıçapı Hertz temas mekaniğine göre genişler
        yaricap = int(np.clip(np.sqrt(normal_kuvvet) * 3.0, 2, self.res // 2 - 2))

        for dy in range(-yaricap, yaricap + 1):
            for dx in range(-yaricap, yaricap + 1):
                r = np.sqrt(dx**2 + dy**2)
                if r <= yaricap:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.res and 0 <= ny < self.res:
                        # Parabolik Hertz temas basınç profili
                        p = (normal_kuvvet / (np.pi * yaricap**2 + 1e-4)) * (1.0 - (r / (yaricap + 1e-4))**2)
                        basinc_haritasi[ny, nx] = max(0.0, float(p))

        temas_alani = int(np.sum(basinc_haritasi > 0.01))
        merkez_baski = float(np.sum(basinc_haritasi))

        return {
            "basinc_haritasi": basinc_haritasi,
            "temas_alani_piksel": temas_alani,
            "toplam_baski_kuvveti": round(merkez_baski, 3),
            "temas_yaricapi": yaricap,
        }


class WristForceTorqueSensor:
    """6-Eksenli Bilek Kuvvet ve Tork (F/T) Wrench Ölçüm Modeli."""

    @classmethod
    def read_wrench(cls, nesne_kutlesi_kg: float, ivme_m_s2: float = 0.0) -> np.ndarray:
        """Yerçekimi ve dinamik ivmelenme altındaki 6-DoF kuvvet/tork vektörünü döner."""
        g = 9.81
        Fz = nesne_kutlesi_kg * (g + ivme_m_s2)  # Düşey yerçekimi yükü
        Fx = float(np.random.normal(0.0, 0.05))  # Yanal titreşim
        Fy = float(np.random.normal(0.0, 0.05))
        Tx = Fy * 0.05  # Kol moment kolu (5cm)
        Ty = Fx * 0.05
        Tz = float(np.random.normal(0.0, 0.01))

        return np.array([round(Fx, 3), round(Fy, 3), round(Fz, 3), round(Tx, 4), round(Ty, 4), round(Tz, 4)])


class SlipDetectorAndGraspController:
    """Sürtünme Konisi Marjini ve 1000Hz Dinamik Kavrama Kuvveti Adaptasyonu."""

    def __init__(self, statik_surtunme_katsayisi: float = 0.6, max_kirilma_kuvveti_N: float = 12.0):
        self.mu_s = statik_surtunme_katsayisi
        self.F_max_kirilma = max_kirilma_kuvveti_N

    def detect_slip(self, Fn: float, Ft: float) -> Tuple[bool, float]:
        """Teğetsel kuvvetin normal kuvvete oranını sürtünme konisiyle karşılaştırır."""
        if Fn <= 0.05:
            return True, 1.0  # Temas yoksa kayma kesin

        oran = abs(Ft) / Fn
        kayma_var = oran >= (self.mu_s * 0.85)  # %85 marjinde mikro-kayma uyarısı
        return kayma_var, round(float(oran), 3)

    def compute_adaptive_force(self, mevcut_Fn: float, Ft: float, is_fragile: bool = True) -> float:
        """Kayma tespit edildiğinde kuvveti kademeli artırır, kırılma tavanını aşmaz."""
        kayma_var, oran = self.detect_slip(mevcut_Fn, Ft)

        if kayma_var:
            # Kaymayı durdurmak için gereken minimum normal kuvvet + güvenlik payı
            gerekli_Fn = (abs(Ft) / (self.mu_s * 0.75)) + 0.5
            yeni_Fn = max(mevcut_Fn + 0.8, gerekli_Fn)
        else:
            # Kararlı tutuşta gereksiz ezmeyi önlemek için hafif gevşetme
            yeni_Fn = max(1.0, mevcut_Fn - 0.05)

        if is_fragile:
            yeni_Fn = min(yeni_Fn, self.F_max_kirilma)

        return round(float(yeni_Fn), 3)


class TactileGraspPipeline:
    """Kırılgan Nesneler İçin Dokunsal Kapalı Döngü Kavrama Simülatörü."""

    def __init__(self):
        self.tactile = GelSightTactileSensor(res=32)
        self.controller = SlipDetectorAndGraspController()

    def simulate_fragile_grasp(self, adim_sayisi: int = 10) -> Dict[str, Any]:
        """Dış sarsıntılar altında kırılgan bir nesneyi tutma döngüsünü simüle eder."""
        Fn = 2.0  # Başlangıç tutuş kuvveti (N)
        gecmis_Fn: List[float] = []
        gecmis_Ft: List[float] = []
        kayma_olaylari: List[bool] = []

        for adim in range(adim_sayisi):
            # Değişken yerçekimi ve ivmelenme yükü (Ft)
            Ft = 1.2 + 0.6 * np.sin(adim * 0.8)
            kayma, _ = self.controller.detect_slip(Fn, Ft)
            Fn = self.controller.compute_adaptive_force(Fn, Ft, is_fragile=True)

            gecmis_Fn.append(Fn)
            gecmis_Ft.append(round(float(Ft), 3))
            kayma_olaylari.append(kayma)

        son_patch = self.tactile.get_contact_patch(Fn)

        return {
            "adim_sayisi": adim_sayisi,
            "gecmis_Fn": gecmis_Fn,
            "gecmis_Ft": gecmis_Ft,
            "kayma_olay_sayisi": sum(kayma_olaylari),
            "son_temas_alani": son_patch["temas_alani_piksel"],
            "kirilma_oldu_mu": max(gecmis_Fn) > self.controller.F_max_kirilma,
            "dusurme_oldu_mu": min(gecmis_Fn) < 0.5,
        }
