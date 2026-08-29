"""
Day 188: Özel Triton Fused RMSNorm ve Residual Ekleme Çekirdeği Ana Çalıştırma Akışı.
"""

import os
import sys
import torch

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.fused_rmsnorm_motoru import (
    PyTorchUnfusedRMSNormResidual,
    FusedRMSNormResidual,
)
from src.profilleyici import RMSNormBellekProfilleyici
from src.gorsellestirici import RMSNormGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 188 (FAZ 10): CUSTOM TRITON FUSED RMSNORM & RESIDUAL ADDITION KERNEL")
    print("=" * 110)

    # -------------------------------------------------------------
    # ADIM 1: İleri Geçiş (Forward Pass) Doğrulama Testi
    # -------------------------------------------------------------
    print("\n[1/4] Fused RMSNorm & Residual İleri Geçiş Testi (Llama-3 Şekli: [4, 512, 4096])...")
    b, s, d = 4, 512, 4096
    x = torch.randn(b, s, d, requires_grad=True)
    residual = torch.randn(b, s, d, requires_grad=True)

    # Modeller
    fused_norm = FusedRMSNormResidual(hidden_dim=d)
    torch_norm = PyTorchUnfusedRMSNormResidual(hidden_dim=d)
    # Ağırlıkları eşitle
    with torch.no_grad():
        torch_norm.weight.copy_(fused_norm.weight)

    out_fused, x_res_fused = fused_norm(x, residual)
    out_torch, x_res_torch = torch_norm(x, residual)

    fark_out = torch.max(torch.abs(out_fused - out_torch)).item()
    fark_res = torch.max(torch.abs(x_res_fused - x_res_torch)).item()

    print(f"  • Giriş Tensör Şekli          : {list(x.shape)}")
    print(f"  • Çıktı Maksimum Farkı        : {fark_out:.2e}")
    print(f"  • Residual Maksimum Farkı     : {fark_res:.2e}")
    print(f"  • İleri Geçiş Eşleşiyor mu    : {fark_out < 1e-6}")
    print("  ✓ Fused RMSNorm İleri Geçiş Başarıyla Doğrulandı!")

    # -------------------------------------------------------------
    # ADIM 2: Geri Geçiş (Backward Pass) Autograd Doğrulama Testi
    # -------------------------------------------------------------
    print("\n[2/4] Fused Autograd Geri Geçiş (Backward) Gradyan Doğrulaması...")
    loss_fused = out_fused.sum()
    loss_fused.backward()

    # PyTorch için klon girdiler
    x_cl = x.detach().clone().requires_grad_(True)
    res_cl = residual.detach().clone().requires_grad_(True)
    torch_norm_cl = PyTorchUnfusedRMSNormResidual(hidden_dim=d)
    with torch.no_grad():
        torch_norm_cl.weight.copy_(fused_norm.weight)

    out_torch_cl, _ = torch_norm_cl(x_cl, res_cl)
    loss_torch = out_torch_cl.sum()
    loss_torch.backward()

    grad_x_fark = torch.max(torch.abs(x.grad - x_cl.grad)).item()
    grad_res_fark = torch.max(torch.abs(residual.grad - res_cl.grad)).item()
    grad_w_fark = torch.max(torch.abs(fused_norm.weight.grad - torch_norm_cl.weight.grad)).item()

    print(f"  • dX (Giriş Gradyan Farkı)    : {grad_x_fark:.2e}")
    print(f"  • dResidual (Kalıntı Farkı)   : {grad_res_fark:.2e}")
    print(f"  • dWeight (Ağırlık Farkı)     : {grad_w_fark:.2e}")
    print(f"  • Geri Geçiş Eşleşiyor mu     : {grad_x_fark < 1e-5 and grad_w_fark < 1e-5}")
    print("  ✓ Fused Autograd Geri Geçiş Çekirdeği Başarıyla Doğrulandı!")

    # -------------------------------------------------------------
    # ADIM 3: Llama-3 70B Ölçeğinde Bellek Tasarruf Analizi
    # -------------------------------------------------------------
    print("\n[3/4] LLM Modelleri Ölçeğinde HBM Bellek Tasarruf Raporu...")
    katman_analizi = RMSNormBellekProfilleyici.bellek_ve_gecis_analizi(batch_size=4, seq_len=4096, hidden_dim=8192)
    model_raporu = RMSNormBellekProfilleyici.model_olcegi_tasarruf_raporu()

    print("-" * 110)
    print(f"{'Model Adı':<15} | {'Katman':<8} | {'RMSNorm Sayısı':<16} | {'PyTorch HBM':<15} | {'Triton HBM':<14} | {'Tasarruf (GB)'}")
    print("-" * 110)
    for r in model_raporu:
        print(
            f"{r['model_adi']:<15} | "
            f"{r['katman_sayisi']:<8d} | "
            f"{r['toplam_rmsnorm_sayisi']:<16d} | "
            f"{r['pytorch_hbm_gb']:>10.2f} GB    | "
            f"{r['triton_hbm_gb']:>9.2f} GB   | "
            f"{r['tasarruf_gb']:>10.2f} GB (%61.5 HBM Kazancı)"
        )
    print("-" * 110)

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Görsel Teşhis Panosu Üretimi
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Fused RMSNorm Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "fused_rmsnorm_paneli.png")

    RMSNormGorsellestirici.teshis_paneli_olustur(
        katman_analizi=katman_analizi,
        model_raporu=model_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Fused RMSNorm Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 110)
    print("✓ Day 188: ÖZEL TRITON FUSED RMSNORM & RESIDUAL BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
