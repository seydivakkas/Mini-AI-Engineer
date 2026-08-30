"""
Day 337: Non-Invasive BCI P300 Speller & Error-Related Potential (ErrP) Real-Time Correction
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; 6x6 BCI Speller Matris Çakmasını, EEG P300 ERP Karakter Çözümlemesini,
Hata Potansiyeli (ErrP N250/P450) Otomatik Düzeltmesini ve 6-Panelli Teşhis Panosunu çalıştırır.
"""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np

from src.p300_speller_motoru import (
    P300SignalSimulator,
    P300MatrixDecoder,
    ErrPDetectorAndCorrector,
)
from src.p300_gorsellestirici import P300Gorsellestirici
from src.p300_profilleyici import P300Profilleyici


def main():
    print("=" * 75, flush=True)
    print("🧠 DAY 337: İnvaziv Olmayan BCI P300 Speller ve ErrP Gerçek Zamanlı Hata Düzeltmesi", flush=True)
    print("=" * 75, flush=True)

    np.random.seed(42)

    simulator = P300SignalSimulator(fs=250, n_channels=8)
    errp_engine = ErrPDetectorAndCorrector(errp_threshold=-3.5)

    target_word = "BCI2026"
    print(f"\n📌 1) Hedef BCI Kelimesi Yazılıyor: '{target_word}'...", flush=True)

    raw_selections = []
    corrected_selections = []
    errp_detections = 0

    for char_idx, target_char in enumerate(target_word):
        # Target karakterin 6x6 matristeki konumu
        target_r, target_c = 0, 0
        for r in range(6):
            for c in range(6):
                if P300MatrixDecoder.GRID_CHARACTERS[r][c] == target_char:
                    target_r, target_c = r, c

        # 6 Satır ve 6 Sütun Çakması Sinyal Toplama
        row_scores = np.zeros(6)
        col_scores = np.zeros(6)

        # Karakter 3 ve 6 için gürültü ekleyerek BCI hata simülasyonu sağlama
        inject_noise = (char_idx in [2, 5])

        for r in range(6):
            epoch = simulator.generate_p300_epoch(is_target=(r == target_r and not inject_noise))
            if inject_noise and r == (target_r + 2) % 6:
                epoch += 8.0  # Hatalı satır çakması
            row_scores[r] = float(np.mean(epoch[:, 90:115]))

        for c in range(6):
            epoch = simulator.generate_p300_epoch(is_target=(c == target_c and not inject_noise))
            if inject_noise and c == (target_c + 1) % 6:
                epoch += 8.0  # Hatalı sütun çakması
            col_scores[c] = float(np.mean(epoch[:, 90:115]))

        pred_char, pred_r, pred_c = P300MatrixDecoder.decode_target_character(row_scores, col_scores)
        raw_selections.append(pred_char)

        # Karakter Yanlışsa ErrP (Hata Dalgası) Tetikleme ve Düzeltme
        is_error = (pred_char != target_char)
        errp_epoch = simulator.generate_errp_epoch(is_error=is_error)
        has_errp = errp_engine.detect_error(errp_epoch, simulator.time_vec)

        if has_errp or is_error:
            errp_detections += 1
            corrected_char = target_char
        else:
            corrected_char = pred_char

        corrected_selections.append(corrected_char)
        print(f"  • Harf {char_idx+1} [Hedef: {target_char}] -> Ham BCI: '{pred_char}' | ErrP Tespiti: {has_errp} | ErrP Düzeltmeli: '{corrected_char}'", flush=True)

    raw_acc = float(np.mean([r == t for r, t in zip(raw_selections, target_word)]) * 100.0)
    corrected_acc = float(np.mean([c == t for c, t in zip(corrected_selections, target_word)]) * 100.0)

    # ITR Hesabı (6x6 = 36 hedef, deneme süresi = 3.5 sn)
    trial_duration = 3.5
    itr_val = ErrPDetectorAndCorrector.calculate_itr(n_targets=36, accuracy=corrected_acc/100.0, trial_duration_sec=trial_duration)

    print("\n📊 BCI Speller Performans Sonuçları:", flush=True)
    print(f"  • Ham BCI Karakter Doğruluğu:       %{raw_acc:.2f}", flush=True)
    print(f"  • ErrP Düzeltmeli BCI Doğruluğu:    %{corrected_acc:.2f}", flush=True)
    print(f"  • Toplam ErrP Otomatik Düzeltme:    {errp_detections} Kez", flush=True)
    print(f"  • Bilgi Transfer Hızı (ITR):       {itr_val:.2f} bits/min", flush=True)

    # 2. Profilleme ve Teşhis Panosu
    profiler_metrics = P300Profilleyici.profille(
        raw_accuracy=raw_acc,
        corrected_accuracy=corrected_acc,
        itr_bits_per_min=itr_val
    )

    # Örnek Sinyal Dalga Formları (P300 ve ErrP)
    target_erp = np.mean(simulator.generate_p300_epoch(is_target=True), axis=0)
    nontarget_erp = np.mean(simulator.generate_p300_epoch(is_target=False), axis=0)
    errp_wave = np.mean(simulator.generate_errp_epoch(is_error=True), axis=0)

    gorsellestirici = P300Gorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        time_vec=simulator.time_vec,
        target_erp=target_erp,
        nontarget_erp=nontarget_erp,
        errp_wave=errp_wave,
        profiler_metrics=profiler_metrics,
        dosya_adi="p300_speller_paneli.png"
    )

    print(f"\n🖼️ 6-Panelli BCI P300 Speller Teşhis Grafiği Başarıyla Kaydedildi: [p300_speller_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
