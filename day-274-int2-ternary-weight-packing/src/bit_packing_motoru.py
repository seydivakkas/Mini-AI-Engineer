"""
Day 274 (FAZ 14): Bit Düzeyinde Paketleme (Bit-Packing) ve Çözme (Unpacking) Motoru.
2-Bit (INT2) ve 1.58-Bit (Ternary {-1, 0, 1}) Ağırlıkların UINT32 İçinde Sıkıştırılması ve Fused GEMM Çekirdeği.
"""

from typing import Dict, Any, Tuple
import numpy as np


class BitPackingKernelEngine:
    """
    2-Bit / Ternary Ağırlık Paketleme (Bit-Packing) ve Donanım Çözme Motoru.
    
    Özellikler:
    - 16 adet 2-bit ağırlığın tek bir 32-bit UINT32 tam sayısında paketlenmesi
    - Ternary {-1, 0, +1} -> {0, 1, 2} bias eşlemesi ve bit kaydırma
    - Donanım seviyesi SIMD bit maskeleme ve çözme simülasyonu
    - Fused Unpack + GEMM (Kayıpsız ve Doğrudan Register Üzerinde İşlem)
    - %87.5 VRAM tasarrufu ve 8.0x bellek bant genişliği kazanımı
    """

    PACKING_FACTOR_2BIT = 16  # 32 bit / 2 bit = 16 eleman / uint32

    @classmethod
    def pack_int2_weights(cls, weights: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
        """
        [0, 3] aralığındaki 2-bit ağırlıkları UINT32 dizisine paketler.
        """
        orig_shape = weights.shape
        flat = weights.flatten().astype(np.uint32)
        total_elements = len(flat)

        # 16'nın katına tamamla (padding)
        pad_len = (cls.PACKING_FACTOR_2BIT - (total_elements % cls.PACKING_FACTOR_2BIT)) % cls.PACKING_FACTOR_2BIT
        if pad_len > 0:
            flat = np.pad(flat, (0, pad_len), mode="constant", constant_values=0)

        num_packed = len(flat) // cls.PACKING_FACTOR_2BIT
        packed = np.zeros(num_packed, dtype=np.uint32)

        # 16 elemanı bit kaydırma ile uint32 içine yerleştir
        for i in range(cls.PACKING_FACTOR_2BIT):
            chunk = flat[i::cls.PACKING_FACTOR_2BIT] & 0x3
            packed |= (chunk << (i * 2))

        return packed, orig_shape

    @classmethod
    def unpack_int2_weights(cls, packed: np.ndarray, orig_shape: Tuple[int, ...]) -> np.ndarray:
        """
        UINT32 dizisindeki 2-bit ağırlıkları çözerek orijinal tensörü yeniden oluşturur.
        """
        total_elements = int(np.prod(orig_shape))
        num_packed = len(packed)
        unpacked_flat = np.zeros(num_packed * cls.PACKING_FACTOR_2BIT, dtype=np.uint8)

        for i in range(cls.PACKING_FACTOR_2BIT):
            unpacked_flat[i::cls.PACKING_FACTOR_2BIT] = (packed >> (i * 2)) & 0x3

        unpacked = unpacked_flat[:total_elements].reshape(orig_shape)
        return unpacked

    @classmethod
    def pack_ternary_weights(cls, weights: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
        """
        {-1, 0, 1} ternary ağırlıklarını {0, 1, 2} olarak offsetleyip UINT32 içine paketler.
        """
        shifted = (weights + 1).astype(np.uint32)
        return cls.pack_int2_weights(shifted)

    @classmethod
    def unpack_ternary_weights(cls, packed: np.ndarray, orig_shape: Tuple[int, ...]) -> np.ndarray:
        """
        UINT32 dizisinden ternary ağırlıkları çözer ve [-1, 0, 1] aralığına geri döndürür.
        """
        unpacked_shifted = cls.unpack_int2_weights(packed, orig_shape).astype(np.int8)
        return unpacked_shifted - 1

    @classmethod
    def fused_packed_gemm(
        cls,
        x: np.ndarray,
        packed_weights: np.ndarray,
        weight_shape: Tuple[int, int],
        scale: float = 1.0,
    ) -> np.ndarray:
        """
        Donanım Fused Unpack + GEMM Çekirdeği: Ağırlıklar VRAM'den uint32 olarak okunur,
        GPU register içinde anlık çözülüp matris çarpımı yapılır.
        """
        # x: (Batch, K), weight_shape: (K, N)
        unpacked_w = cls.unpack_ternary_weights(packed_weights, weight_shape).astype(np.float32)
        # Y = X * (W * scale)
        y = np.dot(x, unpacked_w * scale)
        return y

    @classmethod
    def execute_mock_packing_pipeline(
        cls,
        matrix_rows: int = 4096,
        matrix_cols: int = 4096,
    ) -> Dict[str, Any]:
        """
        4096 x 4096 ağırlık matrisinde uçtan uca paketleme, bellek ölçümü ve doğrulama.
        """
        np.random.seed(42)
        # {-1, 0, 1} rastgele ternary ağırlık matrisi
        ternary_w = np.random.choice([-1, 0, 1], size=(matrix_rows, matrix_cols)).astype(np.int8)
        
        # 1. Paketleme
        packed_w, shape = cls.pack_ternary_weights(ternary_w)

        # 2. Çözme
        recovered_w = cls.unpack_ternary_weights(packed_w, shape)
        tam_eslesme = np.array_equal(ternary_w, recovered_w)

        # 3. Bellek Boyutları (Byte)
        fp16_bytes = matrix_rows * matrix_cols * 2
        int8_bytes = matrix_rows * matrix_cols * 1
        packed_bytes = packed_w.nbytes  # 2-bit / 16 eleman = 0.25 byte/eleman

        # 4. Fused GEMM Testi
        x = np.random.randn(32, matrix_rows).astype(np.float32)
        out = cls.fused_packed_gemm(x, packed_w, shape, scale=0.05)

        return {
            "matrix_shape": (matrix_rows, matrix_cols),
            "fp16_mb": fp16_bytes / (1024.0 * 1024.0),
            "int8_mb": int8_bytes / (1024.0 * 1024.0),
            "packed_mb": packed_bytes / (1024.0 * 1024.0),
            "tasarruf_orani_fp16": fp16_bytes / packed_bytes,
            "tam_eslesme": tam_eslesme,
            "gemm_out_shape": out.shape,
        }
