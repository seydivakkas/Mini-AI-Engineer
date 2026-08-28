"""
INT8 Post-Training Kuantizasyon Motoru (PTQ Engine)
===================================================
ONNX modellerini 32-bit kayan noktadan (FP32) 8-bit tam sayiya (INT8)
donusturerek model boyutunu %75 kucultur ve bellek bant genisligi darbogazini cozer.
"""

from typing import Dict, Any, Optional
import os
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType


class INT8Kuantizator:
    """
    ONNX modelleri icin dinamik Post-Training Quantization (PTQ) motoru.
    """

    def __init__(self, agirlik_tipi: QuantType = QuantType.QInt8) -> None:
        self.agirlik_tipi = agirlik_tipi

    def dinamik_kuantize_et(
        self,
        girdi_onnx_yolu: str,
        cikti_int8_yolu: str,
        optimize_et: bool = True
    ) -> Dict[str, Any]:
        """
        FP32 ONNX modelini INT8 dinamik kuantizasyona tabi tutar.

        Args:
            girdi_onnx_yolu: FP32 .onnx dosyasinin yolu
            cikti_int8_yolu: Uretilecek INT8 .onnx dosyasinin yolu
            optimize_et: Model on-optimizasyondan gecirilsin mi

        Returns:
            Kuantizasyon metrikleri ve boyut tasarruf oranlari
        """
        if not os.path.exists(girdi_onnx_yolu):
            raise FileNotFoundError(f"Girdi ONNX modeli bulunamadi: {girdi_onnx_yolu}")

        os.makedirs(os.path.dirname(os.path.abspath(cikti_int8_yolu)), exist_ok=True)

        # ONNX Runtime Dinamik Kuantizasyon API'si
        quantize_dynamic(
            model_input=girdi_onnx_yolu,
            model_output=cikti_int8_yolu,
            weight_type=self.agirlik_tipi,
            extra_options={"EnableSubgraph": True}
        )

        fp32_boyut_mb = os.path.getsize(girdi_onnx_yolu) / (1024 * 1024)
        int8_boyut_mb = os.path.getsize(cikti_int8_yolu) / (1024 * 1024)
        sikistirma_orani = fp32_boyut_mb / (int8_boyut_mb + 1e-9)
        tasarruf_yuzdesi = (1.0 - (int8_boyut_mb / (fp32_boyut_mb + 1e-9))) * 100.0

        int8_model = onnx.load(cikti_int8_yolu)
        dugum_sayisi = len(int8_model.graph.node)
        kuantize_oplar = [node.op_type for node in int8_model.graph.node if "Integer" in node.op_type or "MatMulInteger" in node.op_type or "QLinear" in node.op_type or "DynamicQuantize" in node.op_type]

        return {
            "girdi_fp32_yolu": girdi_onnx_yolu,
            "cikti_int8_yolu": cikti_int8_yolu,
            "fp32_boyut_mb": round(fp32_boyut_mb, 3),
            "int8_boyut_mb": round(int8_boyut_mb, 3),
            "sikistirma_orani": round(sikistirma_orani, 2),
            "tasarruf_yuzdesi": round(tasarruf_yuzdesi, 2),
            "int8_dugum_sayisi": dugum_sayisi,
            "kuantize_operator_sayisi": len(kuantize_oplar)
        }
