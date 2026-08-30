"""
Day 337: Non-Invasive BCI P300 Speller & Error-Related Potential (ErrP) Real-Time Correction
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; 8-Kanal EEG P300/ErrP Sinyal Simülatörünü, 6x6 BCI Speller Matris Çözücüyü
ve Hata Potansiyeli (ErrP) Otomatik Düzeltme Motorunu içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import math
import numpy as np


class P300SignalSimulator:
    """
    8-Kanallı EEG Olaya İlişkin Potansiyel (ERP) Sinyal Simülatörü.
    P300 pozitif tepe dalgası (+10uV @ 300ms) ve ErrP negatif dalgası (-8uV @ 250ms) üretir.
    """
    def __init__(self, fs: int = 250, n_channels: int = 8):
        self.fs = fs
        self.n_channels = n_channels
        self.time_vec = np.linspace(-0.1, 0.8, int(fs * 0.9))  # -100ms ile 800ms arası (225 zaman noktası)

    def generate_p300_epoch(self, is_target: bool) -> np.ndarray:
        """
        Target veya Non-Target çakması için 8-kanallı EEG epoku üretir.
        """
        noise = np.random.normal(0, 2.5, size=(self.n_channels, len(self.time_vec)))
        
        if is_target:
            # P300 dalgası: t = 300ms (0.3s) civarında pozitif tepe
            p300_wave = 10.0 * np.exp(-((self.time_vec - 0.30) ** 2) / (2 * (0.04 ** 2)))
            for ch in range(self.n_channels):
                noise[ch] += p300_wave * (0.8 + 0.4 * np.random.rand())
        
        return noise

    def generate_errp_epoch(self, is_error: bool) -> np.ndarray:
        """
        Karakter seçimi sonrası Hata Potansiyeli (ErrP) epoku üretir.
        N250 (250ms negatif) + P450 (450ms pozitif) bileşeni barındırır.
        """
        noise = np.random.normal(0, 2.0, size=(self.n_channels, len(self.time_vec)))
        
        if is_error:
            # ErrP dalgası: N250 (-8 uV @ 0.25s) + P450 (+6 uV @ 0.45s)
            n250 = -8.0 * np.exp(-((self.time_vec - 0.25) ** 2) / (2 * (0.03 ** 2)))
            p450 = 6.0 * np.exp(-((self.time_vec - 0.45) ** 2) / (2 * (0.04 ** 2)))
            errp_wave = n250 + p450
            for ch in range(self.n_channels):
                noise[ch] += errp_wave * (0.8 + 0.4 * np.random.rand())

        return noise


class P300MatrixDecoder:
    """
    6x6 BCI Speller Matrisi ve Satır/Sütun P300 Skorlama Çözücüsü.
    """
    GRID_CHARACTERS = [
        ['A', 'B', 'C', 'D', 'E', 'F'],
        ['G', 'H', 'I', 'J', 'K', 'L'],
        ['M', 'N', 'O', 'P', 'Q', 'R'],
        ['S', 'T', 'U', 'V', 'W', 'X'],
        ['Y', 'Z', '0', '1', '2', '3'],
        ['4', '5', '6', '7', '8', '9']
    ]

    def __init__(self):
        pass

    @staticmethod
    def decode_target_character(row_scores: np.ndarray, col_scores: np.ndarray) -> Tuple[str, int, int]:
        """
        Satır ve sütun P300 skorlarından en yüksek olasılıklı karakteri tahmin eder.
        """
        best_row = int(np.argmax(row_scores))
        best_col = int(np.argmax(col_scores))
        pred_char = P300MatrixDecoder.GRID_CHARACTERS[best_row][best_col]
        return pred_char, best_row, best_col


class ErrPDetectorAndCorrector:
    """
    Hata Potansiyeli (ErrP) Algılama ve Gerçek Zamanlı Otomatik Düzeltme Motoru.
    """
    def __init__(self, errp_threshold: float = -3.5):
        self.errp_threshold = errp_threshold

    def detect_error(self, errp_epoch: np.ndarray, time_vec: np.ndarray) -> bool:
        """
        EEG sinyalinin t=200-300ms aralığındaki ortalama genliğine bakarak ErrP tespit eder.
        """
        mask_250ms = (time_vec >= 0.20) & (time_vec <= 0.30)
        mean_amplitude_250ms = float(np.mean(errp_epoch[:, mask_250ms]))
        return mean_amplitude_250ms < self.errp_threshold

    @staticmethod
    def calculate_itr(n_targets: int, accuracy: float, trial_duration_sec: float) -> float:
        """
        Bilgi Transfer Hızını (Information Transfer Rate - ITR, bits/min) hesaplar.
        ITR = B * [ log2(N) + P*log2(P) + (1-P)*log2((1-P)/(N-1)) ]
        """
        if accuracy <= (1.0 / n_targets):
            return 0.0
        if accuracy >= 0.999:
            accuracy = 0.999

        p = accuracy
        n = float(n_targets)
        
        bits_per_selection = math.log2(n) + p * math.log2(p) + (1.0 - p) * math.log2((1.0 - p) / (n - 1.0))
        selections_per_min = 60.0 / trial_duration_sec
        itr = selections_per_min * bits_per_selection
        return max(0.0, float(itr))
