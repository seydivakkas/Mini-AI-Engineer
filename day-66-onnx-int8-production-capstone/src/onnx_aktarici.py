"""
ONNX Aktarim ve Dogrulama Motoru (ONNX Exporter)
================================================
PyTorch modellerini endustriyel standartta ONNX formatina donusturur,
dinamik eksenleri (Dynamic Axes) yapilandirir ve grafik butunlugunu dogrular.
"""

from typing import Dict, List, Optional, Tuple, Any
import os
import torch
import torch.nn as nn
import onnx
from onnx import shape_inference


class ONNXDonusturucu:
    """
    PyTorch modellerini ONNX formatina donusturen ve dogrulayan sinif.
    """

    def __init__(self, opset_versiyonu: int = 18) -> None:
        self.opset_versiyonu = opset_versiyonu

    def disa_aktar(
        self,
        model: nn.Module,
        ornek_girdi: torch.Tensor,
        cikti_yolu: str,
        girdi_adi: str = "girdi_gorsel",
        cikti_adi: str = "cikis_lojiti",
        dinamik_eksenler: Optional[Dict[str, Dict[int, str]]] = None,
        verbose: bool = False
    ) -> str:
        """
        PyTorch modelini ONNX dosyasina aktarir.

        Args:
            model: Egitilmis PyTorch modeli
            ornek_girdi: Modeli izlemek (tracing) icin ornek tensör
            cikti_yolu: Hedef .onnx dosya yolu
            girdi_adi: ONNX girdi tensorunun ismi
            cikti_adi: ONNX cikti tensorunun ismi
            dinamik_eksenler: Degisken boyutlu eksen yapisi (ornegin batch_size)
            verbose: Ayrintili cikti yazdirilsin mi
        """
        os.makedirs(os.path.dirname(os.path.abspath(cikti_yolu)), exist_ok=True)

        # Modeli degerlendirme moduna al
        model.eval()

        if dinamik_eksenler is None:
            dinamik_eksenler = {
                girdi_adi: {0: "batch_size"},
                cikti_adi: {0: "batch_size"}
            }

        with torch.no_grad():
            torch.onnx.export(
                model,
                ornek_girdi,
                cikti_yolu,
                export_params=True,
                opset_version=self.opset_versiyonu,
                do_constant_folding=True,
                input_names=[girdi_adi],
                output_names=[cikti_adi],
                dynamic_axes=dinamik_eksenler,
                verbose=verbose,
                dynamo=False
            )

        # Grafigin dogrulugunu kontrol et
        self.modeli_dogrula(cikti_yolu)
        return cikti_yolu

    def modeli_dogrula(self, onnx_yolu: str) -> bool:
        """
        ONNX modelinin semantik ve sembolik butunlugunu denetler.
        """
        if not os.path.exists(onnx_yolu):
            raise FileNotFoundError(f"ONNX modeli bulunamadi: {onnx_yolu}")

        onnx_model = onnx.load(onnx_yolu)
        onnx.checker.check_model(onnx_model)

        # Sekil cikarimi (Shape Inference) gerceklestir
        inferred_model = shape_inference.infer_shapes(onnx_model)
        onnx.save(inferred_model, onnx_yolu)
        return True

    def model_ozeti_al(self, onnx_yolu: str) -> Dict[str, Any]:
        """
        ONNX grafiginin mimari detaylarini, dugum sayisini ve girdi/cikti tiplerini dondurur.
        """
        onnx_model = onnx.load(onnx_yolu)
        dugum_sayisi = len(onnx_model.graph.node)
        op_tipleri = set(node.op_type for node in onnx_model.graph.node)

        girdiler = [input.name for input in onnx_model.graph.input]
        ciktilar = [output.name for output in onnx_model.graph.output]
        dosya_boyutu_mb = os.path.getsize(onnx_yolu) / (1024 * 1024)

        return {
            "dosya_yolu": onnx_yolu,
            "opset_versiyonu": onnx_model.opset_import[0].version if onnx_model.opset_import else None,
            "dugum_sayisi": dugum_sayisi,
            "farkli_operatorler": sorted(list(op_tipleri)),
            "girdi_isimleri": girdiler,
            "cikti_isimleri": ciktilar,
            "boyut_mb": round(dosya_boyutu_mb, 3)
        }
