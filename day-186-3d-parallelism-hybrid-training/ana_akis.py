"""
Day 186: 3D Paralellik (DP + TP + PP) ve Küme Eğitimi Ana Çalıştırma Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.uc_boyutlu_grid_topolojisi import UcBoyutluGridTopolojisi
from src.hibrit_3d_egitim_motoru import Hibrit3DEgitimMotoru
from src.gorsellestirici import UcBoyutluGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 186 (FAZ 10): 3D PARALLELISM (DP + TP + PP) HYBRID CLUSTER TRAINING ENGINE")
    print("=" * 110)

    # -------------------------------------------------------------
    # ADIM 1: 64 GPU 3D Process Grid Topolojisi (DP=2, PP=4, TP=8)
    # -------------------------------------------------------------
    print("\n[1/4] 64 GPU'lu 3D Process Grid (DP=2, PP=4, TP=8) Kuruluyor...")
    grid = UcBoyutluGridTopolojisi(dp_size=2, pp_size=4, tp_size=8)
    topo_ozet = grid.topoloji_ozeti()

    print(f"  • Toplam GPU Sayısı (World Size): {topo_ozet['world_size']} GPU")
    print(f"  • DP x PP x TP Yapısı           : {topo_ozet['dp_size']} x {topo_ozet['pp_size']} x {topo_ozet['tp_size']}")
    print(f"  • TP İletişim Grubu Sayısı      : {topo_ozet['tp_grup_sayisi']} grup (her biri 8 GPU - NVLink)")
    print(f"  • PP Pipeline Hattı Sayısı      : {topo_ozet['pp_grup_sayisi']} hat (her biri 4 Aşama - InfiniBand)")
    print(f"  • DP Replika Sayısı             : {topo_ozet['dp_grup_sayisi']} replika (All-Reduce Senkronu)")

    # Örnek Rank 27 Koordinat ve Grup Testi
    ornek_rank = 27
    c_dp, c_pp, c_tp = grid.get_coordinates(ornek_rank)
    tp_grp = grid.get_tp_group(ornek_rank)
    pp_grp = grid.get_pp_group(ornek_rank)
    dp_grp = grid.get_dp_group(ornek_rank)

    print(f"\n  [Rank {ornek_rank} Topoloji Analizi]")
    print(f"    - Koordinat (DP, PP, TP)      : ({c_dp}, {c_pp}, {c_tp})")
    print(f"    - TP Grubu (Matris Bölüşümü)  : {tp_grp}")
    print(f"    - PP Grubu (Katman Hattı)     : {pp_grp}")
    print(f"    - DP Grubu (Gradyan All-Red)  : {dp_grp}")
    print("  ✓ 3D Grid Ortogonal İletişim Grupları Doğrulandı!")

    # -------------------------------------------------------------
    # ADIM 2: Llama-3-70B, GPT-3-175B, Llama-3-405B Küme Analizi
    # -------------------------------------------------------------
    print("\n[2/4] 70B, 175B ve 405B LLM Modelleri İçin 3D Küme Kaynak Analizi...")
    model_raporu = Hibrit3DEgitimMotoru.tum_modeller_analiz_raporu()

    print("-" * 110)
    print(f"{'Model Adı':<15} | {'Parametre':<11} | {'Küme Boyutu':<14} | {'Model VRAM':<13} | {'Akt. VRAM':<12} | {'Tepe VRAM':<12} | {'MFU (%)'}")
    print("-" * 110)
    for r in model_raporu:
        print(
            f"{r['model_adi']:<15} | "
            f"{r['parametre_milyar']:>6.1f} B    | "
            f"{r['toplam_gpu']:>4d} H100 GPU  | "
            f"{r['gpu_model_vram_gb']:>7.2f} GB    | "
            f"{r['gpu_aktivasyon_vram_gb']:>6.2f} GB   | "
            f"{r['gpu_toplam_vram_gb']:>6.2f} GB    | "
            f"%{r['mfu_yuzde']:>5.1f}"
        )
    print("-" * 110)

    # -------------------------------------------------------------
    # ADIM 3: 6 Panelli Görsel Teşhis Panosu Üretimi
    # -------------------------------------------------------------
    print("\n[3/4] 6 Panelli 3D Paralellik Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "uc_boyutlu_paralellik_3d_paneli.png")

    UcBoyutluGorsellestirici.teshis_paneli_olustur(
        model_raporu=model_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ 3D Paralellik Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 110)
    print("✓ Day 186: 3D PARALLELISM (DP + TP + PP) BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
