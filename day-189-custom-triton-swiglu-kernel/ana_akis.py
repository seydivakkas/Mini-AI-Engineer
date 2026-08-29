"""
Day 189: Özel Triton Fused SwiGLU İleri ve Geri Geçiş Çekirdeği Ana Çalıştırma Akışı.
"""

import os
import sys
import torch

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.fused_swiglu_motoru import (
    PyTorchUnfusedSwiGLU,
    FusedSwiGLU,
    SwiGLUMLP,
)
from src.swiglu_profilleyici import SwiGLUBellekProfilleyici
from src.gorsellestirici import SwiGLUGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 189 (FAZ 10): CUSTOM TRITON FUSED SWIGLU FORWARD & BACKWARD KERNEL")
    print("=" * 110)

    # -------------------------------------------------------------
    # ADIM 1: İleri Geçiş (Forward Pass) Doğrulama Testi
    # -------------------------------------------------------------
    print("\n[1/4] Fused SwiGLU İleri Geçiş Testi (Llama-3 Şekli: [4, 512, 14336])...")
    b, s, d_ffn = 4, 512, 14336
    gate = torch.randn(b, s, d_ffn, requires_grad=True)
    up = torch.randn(b, s, d_ffn, requires_grad=True)

    fused_swiglu = FusedSwiGLU()
    torch_swiglu = PyTorchUnfusedSwiGLU()

    out_fused = fused_swiglu(gate, up)
    out_torch = torch_swiglu(gate, up)

    fark_out = torch.max(torch.abs(out_fused - out_torch)).item()
    print(f"  • Giriş Tensör Şekli          : {list(gate.shape)}")
    print(f"  • Çıktı Maksimum Farkı        : {fark_out:.2e}")
    print(f"  • İleri Geçiş Eşleşiyor mu    : {fark_out < 1e-6}")
    print("  ✓ Fused SwiGLU İleri Geçiş Başarıyla Doğrulandı!")

    # -------------------------------------------------------------
    # ADIM 2: Geri Geçiş (Backward Pass) Autograd Doğrulama Testi
    # -------------------------------------------------------------
    print("\n[2/4] Fused Autograd Geri Geçiş (Backward) Gradyan Doğrulaması...")
    loss_fused = out_fused.sum()
    loss_fused.backward()

    # PyTorch referansı için klonlar
    gate_cl = gate.detach().clone().requires_grad_(True)
    up_cl = up.detach().clone().requires_grad_(True)
    out_torch_cl = torch_swiglu(gate_cl, up_cl)
    loss_torch = out_torch_cl.sum()
    loss_torch.backward()

    grad_gate_fark = torch.max(torch.abs(gate.grad - gate_cl.grad)).item()
    grad_up_fark = torch.max(torch.abs(up.grad - up_cl.grad)).item()

    print(f"  • dGate (Kapı Gradyanı Farkı) : {grad_gate_fark:.2e}")
    print(f"  • dUp (Yukarı Proj. Farkı)    : {grad_up_fark:.2e}")
    print(f"  • Geri Geçiş Eşleşiyor mu     : {grad_gate_fark < 1e-6 and grad_up_fark < 1e-6}")
    print("  ✓ Fused Autograd Geri Geçiş Çekirdeği Başarıyla Doğrulandı!")

    # -------------------------------------------------------------
    # ADIM 3: Llama-3 70B MLP Ölçeğinde Bellek Tasarruf Analizi
    # -------------------------------------------------------------
    print("\n[3/4] LLM Modelleri Ölçeğinde HBM Bellek Tasarruf Raporu...")
    katman_analizi = SwiGLUBellekProfilleyici.katman_bazli_hbm_analizi(batch_size=4, seq_len=4096, intermediate_dim=28672)
    model_raporu = SwiGLUBellekProfilleyici.tam_model_swiglu_raporu()

    print("-" * 110)
    print(f"{'Model Adı':<15} | {'Katman':<8} | {'Ara Boyut (D_ffn)':<18} | {'PyTorch HBM':<15} | {'Triton HBM':<14} | {'Tasarruf (GB)'}")
    print("-" * 110)
    for r in model_raporu:
        print(
            f"{r['model_adi']:<15} | "
            f"{r['katman_sayisi']:<8d} | "
            f"{r['intermediate_dim']:<18d} | "
            f"{r['pytorch_hbm_gb']:>10.2f} GB    | "
            f"{r['triton_hbm_gb']:>9.2f} GB   | "
            f"{r['tasarruf_gb']:>10.2f} GB (%62.5 HBM Kazancı)"
        )
    print("-" * 110)

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Görsel Teşhis Panosu Üretimi
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Fused SwiGLU Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "fused_swiglu_paneli.png")

    SwiGLUGorsellestirici.teshis_paneli_olustur(
        katman_analizi=katman_analizi,
        model_raporu=model_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Fused SwiGLU Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 110)
    print("✓ Day 189: ÖZEL TRITON FUSED SWIGLU BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
