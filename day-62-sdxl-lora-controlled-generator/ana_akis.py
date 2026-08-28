"""
Day 62: Stable Diffusion XL (SDXL) + LoRA ile Kontrollü Görsel Üretimi Ana Yürütme Betiği.
"""

import os
import sys
import torch

MEVCUT_DIZIN = os.path.abspath(os.path.dirname(__file__))
if MEVCUT_DIZIN not in sys.path:
    sys.path.insert(0, MEVCUT_DIZIN)

from src.sdxl_lora_motoru import SDXLLoRAMotoru
from src.lora_fuzyon_yoneticisi import LoRAFuzyonYoneticisi
from src.gorsellestirici import SDXLLoRAGorsellestirici


def main():
    print("=" * 95, flush=True)
    print(">>> DAY 62: STABLE DIFFUSION XL (SDXL) + LoRA ILE KONTROLLU GORSEL URETIMI & FUZYON MOTORU", flush=True)
    print("=" * 95, flush=True)

    # 1. SDXL LoRA Motorunun Başlatılması
    d_model = 512
    d_text = 768
    print(f"\n[+] 1. Adım: SDXL Cross-Attention ve LoRA Motoru Başlatılıyor (d_model={d_model}, d_text={d_text})...", flush=True)
    model = SDXLLoRAMotoru(d_model=d_model, d_text=d_text)

    # Adaptörleri ekle
    model.adaptor_ekle("stil_lora", rank=8, alpha=16.0, adapter_agirligi=1.0)
    model.adaptor_ekle("karakter_lora", rank=8, alpha=16.0, adapter_agirligi=0.8)
    model.adaptor_ekle("fotogercekcilik_lora", rank=4, alpha=8.0, adapter_agirligi=0.5)

    # 2. Parametre Verimliliği Analizi
    p_analiz = LoRAFuzyonYoneticisi.parametre_verimlilik_analizi(model)
    print("\n" + "=" * 95, flush=True)
    print(">>> 2. PARAMETRE VERİMLİLİĞİ VE TASARRUF RAPORU", flush=True)
    print("=" * 95, flush=True)
    print(f"• Taban Model Parametre Sayısı   : {p_analiz['taban_parametre_sayisi']:,} (Dondurulmuş)")
    print(f"• LoRA Eğitilebilir Parametreler : {p_analiz['lora_parametre_sayisi']:,} (Rank=8/4)")
    print(f"• Toplam Model Parametreleri     : {p_analiz['toplam_parametre_sayisi']:,}")
    print(f"• LoRA Parametre Oranı           : %{p_analiz['lora_oran_yuzde']:.2f}")
    print(f"• Parametre Tasarruf Oranı       : %{p_analiz['tasarruf_orani_yuzde']:.2f} TASARRUF")

    # 3. LoRA Skalası ve CFG Difüzyon Füzyon Deneyi
    print("\n[+] 3. Adım: LoRA Skala (0.0 - 1.2) ve CFG (3.0 - 12.0) Örnekleme Deneyi Yürütülüyor...", flush=True)
    deney_sonuclari = LoRAFuzyonYoneticisi.calistir_fuzyon_deneyi(
        model=model,
        skala_degerleri=[0.0, 0.4, 0.8, 1.2],
        cfg_degerleri=[3.0, 7.5, 12.0]
    )

    print("\n" + "=" * 95, flush=True)
    print(">>> 4. LoRA SKALA VE LATENT STIL KONTROL METRIKLERI", flush=True)
    print("=" * 95, flush=True)
    print(f"{'LoRA Yapilandirmasi':<24} | {'Skala (lambda)':<14} | {'Tabandan Sapma (Delta z)':<25} | {'Kosinus Benzerlik':<18} | {'Gecikme':<10}")
    print("-" * 98)

    for k, d in deney_sonuclari["skala_analizi"].items():
        print(
            f"{k:<24} | "
            f"{d['skala']:>10.2f} | "
            f"{d['delta_l2_norm']:>22.4f} | "
            f"{d['kosinus_benzerlik']:>18.4f} | "
            f"{d['gecikme_ms']:>7.2f} ms"
        )

    # 4. 6 Panelli Teşhis Panosunun Çizilmesi
    print("\n" + "=" * 95, flush=True)
    print(">>> 5. 6 PANELLİ SDXL + LoRA TEŞHİS VE PERFORMANS PANOSU", flush=True)
    print("=" * 95, flush=True)

    hedef_pano = os.path.join(MEVCUT_DIZIN, "ciktilar", "sdxl_lora_paneli.png")
    cikis_yolu = SDXLLoRAGorsellestirici.panel_ciz(
        deney_sonuclari=deney_sonuclari,
        hedef_path=hedef_pano
    )
    print(f"[+] 6 Panelli Teşhis Panosu Kaydedildi: {os.path.abspath(cikis_yolu)}", flush=True)
    print("=" * 95, flush=True)
    print("DAY 62: STABLE DIFFUSION XL (SDXL) + LoRA KONTROLLÜ ÜRETİM BAŞARIYLA TAMAMLANDI!", flush=True)
    print("=" * 95, flush=True)


if __name__ == "__main__":
    main()
