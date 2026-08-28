"""
Üretim Modeli Girdi Tensör Şeması ve Kural Tanımlayıcı (Tensor Schema & Validation Rules).
"""

from typing import Tuple, List, Optional, Any
import numpy as np


class TensorSekilKurali:
    """Tensör şekil kalıbı ve dinamik boyut kuralı."""

    def __init__(
        self,
        sekil_kalibi: Tuple[int, ...],
        kanal_sirasi: str = "NCHW",
        min_batch: int = 1,
        max_batch: int = 64
    ):
        self.sekil_kalibi = sekil_kalibi
        self.kanal_sirasi = kanal_sirasi
        self.min_batch = min_batch
        self.max_batch = max_batch

    def dogrula(self, gercek_sekil: Tuple[int, ...]) -> Tuple[bool, str]:
        """Tensör şeklini dinamik kurallara göre doğrular."""
        if len(gercek_sekil) != len(self.sekil_kalibi):
            return False, f"Boyut sayısı (Rank) uyuşmazlığı! Beklenen: {len(self.sekil_kalibi)}, Gelen: {len(gercek_sekil)}"

        batch_boyutu = gercek_sekil[0]
        if not (self.min_batch <= batch_boyutu <= self.max_batch):
            return False, f"Batch boyutu limit dışı! Gelen: {batch_boyutu}, İzin verilen: [{self.min_batch}, {self.max_batch}]"

        for i, (beklenen, gelen) in enumerate(zip(self.sekil_kalibi[1:], gercek_sekil[1:]), start=1):
            if beklenen != -1 and beklenen != gelen:
                return False, f"{i}. boyutta uyumsuzluk! Beklenen: {beklenen}, Gelen: {gelen}"

        return True, "Şekil doğrulaması başarılı."


class TensorSemasi:
    """Üretim seviyesinde bir derin öğrenme modelinin girdi tensör sözleşmesi."""

    def __init__(
        self,
        model_adi: str = "ResNet-Vision-Inference",
        beklenen_sekil: Tuple[int, ...] = (-1, 3, 224, 224),
        kanal_sirasi: str = "NCHW",
        gecerli_tipler: Optional[List[Any]] = None,
        deger_araligi: Tuple[float, float] = (-3.5, 3.5),
        max_batch: int = 64,
        sureklilik_sarti: bool = True,
        max_bellek_mb: float = 128.0
    ):
        self.model_adi = model_adi
        self.sekil_kurali = TensorSekilKurali(beklenen_sekil, kanal_sirasi=kanal_sirasi, max_batch=max_batch)
        self.gecerli_tipler = gecerli_tipler or [np.float32, np.float16]
        self.deger_araligi = deger_araligi
        self.sureklilik_sarti = sureklilik_sarti
        self.max_bellek_mb = max_bellek_mb
