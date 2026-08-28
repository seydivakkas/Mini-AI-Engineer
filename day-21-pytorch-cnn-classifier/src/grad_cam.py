"""PyTorch Grad-CAM (Gradient-weighted Class Activation Mapping) Modülü.

Bu modül; PyTorch hook mekanizması yardımıyla modelin son evrişimsel katmanından
aktivasyon haritalarını ve gradyanlarını yakalayarak modelin hangi uzamsal bölgelere
odaklanarak karar verdiğini gösteren ısı haritaları (Heatmap / Grad-CAM) üretir (XAI).
"""

from typing import Optional, Tuple
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    """PyTorch modelleri için kanca (hook) tabanlı Grad-CAM açıklanabilirlik sınıfı."""

    def __init__(self, model: nn.Module, hedef_katman: nn.Module) -> None:
        """GradCAM nesnesini ilklendirir ve ileri/geri kancalarını bağlar."""
        self.model = model
        self.hedef_katman = hedef_katman

        self.aktivasyonlar: Optional[torch.Tensor] = None
        self.gradyanlar: Optional[torch.Tensor] = None

        # Kancaları kaydet
        self.hedef_katman.register_forward_hook(self._forward_hook)
        self.hedef_katman.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor) -> None:
        self.aktivasyonlar = output.detach()

    def _backward_hook(
        self, module: nn.Module, grad_input: Tuple[torch.Tensor], grad_output: Tuple[torch.Tensor]
    ) -> None:
        self.gradyanlar = grad_output[0].detach()

    def isi_haritasi_uret(
        self,
        girdi_tensor: torch.Tensor,
        hedef_sinif: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """Verilen girdi tensörü ve hedef sınıf için Grad-CAM ısı haritası üretir.

        Args:
            girdi_tensor: (1, C, H, W) PyTorch tensörü.
            hedef_sinif: Açıklanacak sınıf indeksi (None ise en yüksek tahmin seçilir).

        Returns:
            Tuple[np.ndarray, int]: (H, W) [0.0, 1.0] ısı haritası ve açıklanan sınıf indeksi.
        """
        self.model.eval()
        self.model.zero_grad()

        girdi = girdi_tensor.clone().requires_grad_(True)
        logits = self.model(girdi)

        if hedef_sinif is None:
            hedef_sinif = int(torch.argmax(logits, dim=1).item())

        hedef_skor = logits[0, hedef_sinif]
        hedef_skor.backward()

        # Gradyanların uzamsal ortalamasını alarak kanal ağırlıklarını (alpha_k) hesapla
        # gradyanlar: (1, Channels, H_feat, W_feat)
        agirliklar = torch.mean(self.gradyanlar, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Ağırlıklı aktivasyon haritalarının toplamı: sum_k (alpha_k * A_k)
        cam = torch.sum(agirliklar * self.aktivasyonlar, dim=1, keepdim=True)  # (1, 1, H_feat, W_feat)
        cam = torch.relu(cam)  # Pozitif katkıyı koru (ReLU)

        cam_np = cam.squeeze().cpu().numpy()

        # Orijinal girdi boyutuna yeniden boyutlandır
        _, _, H, W = girdi_tensor.shape
        cam_resized = cv2.resize(cam_np, (W, H))

        # [0.0, 1.0] aralığına normalize et
        min_val = cam_resized.min()
        max_val = cam_resized.max()
        if max_val - min_val > 1e-7:
            cam_norm = (cam_resized - min_val) / (max_val - min_val)
        else:
            cam_norm = np.zeros_like(cam_resized)

        return cam_norm, hedef_sinif

    def bindirme_ciz(
        self,
        orijinal_rgb: np.ndarray,
        isi_haritasi: np.ndarray,
        sinif_adi: str,
        alfa: float = 0.5,
    ) -> plt.Figure:
        """Orijinal görsel, Isı Haritası ve Bindirmeyi 3 panelli bir figür olarak çizer."""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=130)

        axes[0].imshow(orijinal_rgb)
        axes[0].set_title("Orijinal Görsel", fontsize=10, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(isi_haritasi, cmap="jet")
        axes[1].set_title("Grad-CAM Isı Haritası", fontsize=10, fontweight="bold")
        axes[1].axis("off")

        # Isı haritasını Jet renk uzayına çevir ve bindir
        heatmap_uint8 = np.uint8(255 * isi_haritasi)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0

        overlay = (1.0 - alfa) * orijinal_rgb + alfa * heatmap_color_rgb
        overlay = np.clip(overlay, 0.0, 1.0)

        axes[2].imshow(overlay)
        axes[2].set_title(f"Grad-CAM Bindirme (Sınıf: {sinif_adi})", fontsize=10, fontweight="bold")
        axes[2].axis("off")

        plt.tight_layout()
        return fig
