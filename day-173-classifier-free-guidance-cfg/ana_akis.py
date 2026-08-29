"""
Day 173: Classifier-Free Guidance (CFG) ve DDIM Hızlı Örnekleme Zamanlayıcıları Ana Akışı (FAZ 9).
İstem Uyumu Ölçeklendirme (w), DDIM Deterministik ODE Yörüngesi ve Teşhis Panosu.
"""

import os
import sys
import torch

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.cfg_yoneticisi import CFGYoneticisi
from src.ddim_zamanlayici import DDIMZamanlayici
from src.cfg_ddim_evaluator import CFGDualEvaluator
from src.gorsellestirici import CFGGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 173 (FAZ 9): CLASSIFIER-FREE GUIDANCE (CFG) & DETERMINISTIC DDIM FAST SAMPLING (20-50 STEPS)")
    print("=" * 110)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. CFG GÜRÜLTÜ EKSTRAPOLASYONU VE DİNAMİK EŞİKLEME
    # -------------------------------------------------------------
    print("\n[1/3] Classifier-Free Guidance (CFG) Yöneticisi Başlatılıyor...")
    cfg = CFGYoneticisi(varsayilan_guidance_scale=7.5)

    eps_uncond = torch.randn(1, 4, 16, 16)
    eps_cond = eps_uncond + 0.8 * torch.randn(1, 4, 16, 16)  # Koşullu gürültü

    eps_guided = cfg.yonlendirilmis_gurultu_hesapla(
        eps_uncond, eps_cond, guidance_scale=7.5, dinamik_esikleme=True
    )
    print(f"  • Koşulsuz Gürültü (eps_uncond) : {list(eps_uncond.shape)} [Prompt Yok]")
    print(f"  • Koşullu Gürültü (eps_cond)     : {list(eps_cond.shape)} [Prompt Koşullu]")
    print(f"  • CFG Yönlendirilmiş Gürültü     : {list(eps_guided.shape)} [w=7.5 Altın Oran]")

    # -------------------------------------------------------------
    # 2. DDIM 20 ADIMLI HIZLI ÖRNEKLEME SİMÜLASYONU
    # -------------------------------------------------------------
    print("\n[2/3] Deterministik DDIM (eta=0.0) 20-Adım Örneklemesi Başlatılıyor...")
    ddim = DDIMZamanlayici(num_train_timesteps=1000, num_inference_steps=20, eta=0.0)
    z_t = torch.randn(1, 4, 16, 16)  # Saf gürültü z_T

    for step_idx in range(len(ddim.timesteps)):
        z_t = ddim.ornekleme_adimi(z_t, eps_guided, t_idx=step_idx)

    print(f"  • 20 DDIM Adımı Sonunda Elde Edilen Temiz z_0 Tensörü: {list(z_t.shape)}")

    rapor = CFGDualEvaluator.cfg_olcek_analizini_getir()
    z_data = rapor["zamanlayici_kiyaslamasi"]

    print("\n" + "-" * 80)
    print(f"{'Örnekleme Zamanlayıcısı':<35} | {'Adım Sayısı':<15} | {'İnferans Süresi'}")
    print("-" * 80)
    print(f"{'Klasik DDPM (Markov Zinciri)':<35} | {z_data['ddpm_adim']:<15} | {z_data['ddpm_sure_sn']} saniye")
    print(f"{'Deterministik DDIM (ODE)':<35} | {z_data['ddim_adim']:<15} | {z_data['ddim_sure_sn']} saniye ({z_data['hizlanma_faktoru']}x Hızlı)")
    print("-" * 80)

    # -------------------------------------------------------------
    # 3. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli CFG & DDIM Teşhis Panosu Üretiliyor...")
    gorsellestirici = CFGGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "cfg_ddim_paneli.png")
    gorsellestirici.pano_olustur(rapor, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 110)
    print("✓ Day 173: CLASSIFIER-FREE GUIDANCE & DDIM BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
