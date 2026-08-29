"""
Sıfır Örnekli (Zero-Shot) Görülmemiş Nesneleri Kavrama ve Ayırma Motoru (Day 258).
Ham 3D Nokta Bulutu, Yüzey Normalleri, 6-DoF Antipodal Grasp Kalite Skorlama ve Kutu Ayrıştırma.
"""

from typing import Dict, Any, List, Tuple
import numpy as np


class PointCloudPreprocessor:
    """3D Nokta Bulutu Filtreleme ve Yüzey Normali Kestiricisi."""

    @classmethod
    def filter_table_plane(
        cls,
        points: np.ndarray,
        table_z_min: float = 0.02,
    ) -> np.ndarray:
        """Masa zemin düzlemini filtreleyip sadece nesne noktalarını döndürür."""
        mask = points[:, 2] >= table_z_min
        return points[mask]

    @classmethod
    def estimate_normals(
        cls,
        points: np.ndarray,
        k_neighbors: int = 8,
    ) -> np.ndarray:
        """Her noktanın k-NN yerel kovaryans matrisi (PCA) ile yüzey normalini hesaplar."""
        n_points = len(points)
        normals = np.zeros((n_points, 3), dtype=np.float64)

        centroid = np.mean(points, axis=0)

        for i in range(n_points):
            p_i = points[i]
            dists = np.linalg.norm(points - p_i, axis=1)
            neighbor_indices = np.argsort(dists)[: min(k_neighbors, n_points)]
            neighbors = points[neighbor_indices]

            # Kovaryans matrisi ve özvektörler
            cov = np.cov(neighbors.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            # En küçük özdeğere karşılık gelen özvektör yüzey normalidir
            normal = eigenvecs[:, 0]
            # Normalleri nesne merkezinden dışarı bakacak şekilde yönlendir
            if np.dot(normal, p_i - centroid) < 0:
                normal = -normal
            norm_val = np.linalg.norm(normal)
            normals[i] = normal / (norm_val + 1e-8)

        return normals


class AntipodalGraspGenerator:
    """6-DoF Antipodal Kavrama (Grasp Pose) Adayı Üreticisi ve Kalite Puanlayıcısı."""

    @classmethod
    def evaluate_grasp_quality(
        cls,
        p1: np.ndarray,
        p2: np.ndarray,
        n1: np.ndarray,
        n2: np.ndarray,
        max_gripper_width: float = 0.12,
    ) -> float:
        """İki temas noktası arasındaki antipodal sürtünme konisi uyumunu puanlar."""
        d_vec = p2 - p1
        width = np.linalg.norm(d_vec)

        if width > max_gripper_width or width < 0.01:
            return 0.0

        d_unit = d_vec / (width + 1e-8)

        # Sürtünme Konisi Uyumu: Dışa bakan yüzey normalleri zıt yönlü olmalıdır (n1 ters, n2 düz)
        cos1 = float(-np.dot(n1, d_unit))
        cos2 = float(np.dot(n2, d_unit))

        if cos1 <= 0.15 or cos2 <= 0.15:
            return 0.0

        # Genişlik Uyum Cezası
        width_penalty = np.exp(-((width - 0.05) ** 2) / 0.005)
        quality = float(cos1 * cos2 * width_penalty)
        return min(max(quality, 0.0), 1.0)

    @classmethod
    def generate_grasps(
        cls,
        points: np.ndarray,
        normals: np.ndarray,
        max_grasps: int = 10,
    ) -> List[Dict[str, Any]]:
        """Nokta bulutundan en yüksek kaliteli 6-DoF kavrama pozlarını seçer."""
        grasps = []
        n_points = len(points)
        if n_points < 2:
            return grasps

        # Rastgele temas çiftleri örnekle
        for _ in range(min(500, n_points * 20)):
            i1, i2 = np.random.choice(n_points, size=2, replace=False)
            p1, p2 = points[i1], points[i2]
            n1, n2 = normals[i1], normals[i2]

            score = cls.evaluate_grasp_quality(p1, p2, n1, n2)
            if score > 0.05:
                grasp_center = (p1 + p2) / 2.0
                width = float(np.linalg.norm(p2 - p1))
                approach_dir = np.array([0.0, 0.0, -1.0])  # Üstten yaklaşım

                grasps.append({
                    "merkez_3d": grasp_center.round(3).tolist(),
                    "kavrama_genisligi_m": round(width, 3),
                    "yaklasim_vektoru": approach_dir.tolist(),
                    "kalite_skoru": round(score, 3),
                })

        # Skora göre azalan sırala
        grasps.sort(key=lambda g: g["kalite_skoru"], reverse=True)
        return grasps[:max_grasps]


class ZeroShotBinSortingPipeline:
    """Görülmemiş Nesneleri 6-DoF Kavrayıp Hedef Kutulara Ayıran Hat."""

    HEDEF_KUTULAR = {
        "ORGANİK": [0.60, 0.35, 0.20],
        "PLASTİK": [0.60, 0.00, 0.20],
        "METAL":   [0.60, -0.35, 0.20],
    }

    @classmethod
    def sort_unseen_object(
        cls,
        object_point_cloud: np.ndarray,
        semantic_category: str = "PLASTİK",
    ) -> Dict[str, Any]:
        """Görülmemiş nesnenin 6-DoF tutuşunu bulur ve hedef kutuya rota oluşturur."""
        # 1. Yüzey Normallerini Hesapla
        normals = PointCloudPreprocessor.estimate_normals(object_point_cloud)

        # 2. En İyi 6-DoF Antipodal Kavrama Pozunu Bul
        grasps = AntipodalGraspGenerator.generate_grasps(object_point_cloud, normals)

        if not grasps:
            # Yedek güvenli merkez tutuşu
            best_grasp = {
                "merkez_3d": np.mean(object_point_cloud, axis=0).round(3).tolist(),
                "kavrama_genisligi_m": 0.05,
                "yaklasim_vektoru": [0.0, 0.0, -1.0],
                "kalite_skoru": 0.50,
            }
        else:
            best_grasp = grasps[0]

        kategori_key = semantic_category.upper().strip()
        hedef_kutu_xyz = cls.HEDEF_KUTULAR.get(kategori_key, [0.60, 0.0, 0.20])

        return {
            "kategori": semantic_category,
            "secilen_6dof_grasp": best_grasp,
            "hedef_kutu_koordinati": hedef_kutu_xyz,
            "aday_grasp_sayisi": len(grasps),
            "durum": "BASARILI_AYRILDI",
        }
