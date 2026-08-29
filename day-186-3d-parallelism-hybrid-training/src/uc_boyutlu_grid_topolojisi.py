"""
3D Grid Süreç Topolojisi ve Ortogonal İletişim Grupları Motoru (Day 186 - FAZ 10).
DP x TP x PP Süreç Matrisi ve Rank Koordinat Eşlemesi.
"""

from typing import List, Tuple, Dict, Any


class UcBoyutluGridTopolojisi:
    """
    3 Boyutlu Paralellik Süreç Grid'i (3D Process Grid).
    Global Rank <-> (DP, PP, TP) 3D Koordinat Dönüşümü ve İletişim Grupları.
    Toplam GPU Sayısı: N = DP_size * PP_size * TP_size
    """

    def __init__(self, dp_size: int = 2, pp_size: int = 4, tp_size: int = 8):
        assert dp_size >= 1, "DP derecesi en az 1 olmalıdır."
        assert pp_size >= 1, "PP derecesi en az 1 olmalıdır."
        assert tp_size >= 1, "TP derecesi en az 1 olmalıdır."

        self.dp_size = dp_size
        self.pp_size = pp_size
        self.tp_size = tp_size
        self.world_size = dp_size * pp_size * tp_size

    def get_coordinates(self, rank: int) -> Tuple[int, int, int]:
        """
        Global rank (r) -> (dp_rank, pp_rank, tp_rank) 3D koordinat eşlemesi.
        Düzen: (DP, PP, TP)
        """
        assert 0 <= rank < self.world_size, f"Geçersiz rank: {rank}, world_size: {self.world_size}"
        dp_rank = rank // (self.pp_size * self.tp_size)
        remainder = rank % (self.pp_size * self.tp_size)
        pp_rank = remainder // self.tp_size
        tp_rank = remainder % self.tp_size
        return (dp_rank, pp_rank, tp_rank)

    def get_rank(self, dp_rank: int, pp_rank: int, tp_rank: int) -> int:
        """(dp_rank, pp_rank, tp_rank) 3D koordinat -> Global rank eşlemesi."""
        assert 0 <= dp_rank < self.dp_size
        assert 0 <= pp_rank < self.pp_size
        assert 0 <= tp_rank < self.tp_size
        return dp_rank * (self.pp_size * self.tp_size) + pp_rank * self.tp_size + tp_rank

    def get_tp_group(self, rank: int) -> List[int]:
        """Aynı DP ve PP aşamasında olan, matrisi bölen TP grubundaki rank listesi."""
        dp_r, pp_r, _ = self.get_coordinates(rank)
        return [self.get_rank(dp_r, pp_r, t) for t in range(self.tp_size)]

    def get_pp_group(self, rank: int) -> List[int]:
        """Aynı DP ve TP diliminde olan, ardışık katmanları oluşturan PP hattındaki rank listesi."""
        dp_r, _, tp_r = self.get_coordinates(rank)
        return [self.get_rank(dp_r, p, tp_r) for p in range(self.pp_size)]

    def get_dp_group(self, rank: int) -> List[int]:
        """Aynı PP aşaması ve TP diliminde olan, gradyan All-Reduce yapacak DP grubundaki rank listesi."""
        _, pp_r, tp_r = self.get_coordinates(rank)
        return [self.get_rank(d, pp_r, tp_r) for d in range(self.dp_size)]

    def topoloji_ozeti(self) -> Dict[str, Any]:
        """Topolojiye dair özet metrikler."""
        return {
            "dp_size": self.dp_size,
            "pp_size": self.pp_size,
            "tp_size": self.tp_size,
            "world_size": self.world_size,
            "tp_grup_sayisi": self.dp_size * self.pp_size,
            "pp_grup_sayisi": self.dp_size * self.tp_size,
            "dp_grup_sayisi": self.pp_size * self.tp_size,
        }
