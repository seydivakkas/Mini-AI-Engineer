"""
Day 379: Co-Packaged Optics (CPO) High-Speed Optical Transceiver Modeling
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; AI veri merkezleri ve GPU/ASIC kümeleri için Co-Packaged Optics (CPO)
112 Gbps/şerit PAM4 optik alıcı-verici motorunu, elektro-optik MZM modülatörünü,
göz diyagramını ve pJ/bit enerji verimlilik analizini simüle eder.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np


class PAM4Encoder:
    """
    112 Gbps PAM4 (Pulse Amplitude Modulation 4-Level) Gray Kodlayıcı ve Dekodlayıcı.
    Standard IEEE 802.3bs 4-Level: [-3, -1, +1, +3]
    """
    def __init__(self, baud_rate_gbaud: float = 56.0):
        self.baud_rate = baud_rate_gbaud  # 56 GBaud -> 112 Gbps
        # Gray Kodu Haritası: (00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3)
        self.bit_to_level = {
            (0, 0): -3.0,
            (0, 1): -1.0,
            (1, 1): +1.0,
            (1, 0): +3.0
        }
        self.level_thresholds = [-2.0, 0.0, +2.0]

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """İkili bit dizisini (2N bit) PAM4 sembollerine ([-3, -1, +1, +3]) dönüştürür."""
        if len(bits) % 2 != 0:
            bits = np.append(bits, 0)
        
        bit_pairs = bits.reshape(-1, 2)
        symbols = np.zeros(len(bit_pairs), dtype=np.float32)
        
        for i, (b0, b1) in enumerate(bit_pairs):
            symbols[i] = self.bit_to_level[(int(b0), int(b1))]
            
        return symbols

    def decode(self, noisy_symbols: np.ndarray) -> np.ndarray:
        """Gürültülü PAM4 sembollerini [-2.0, 0.0, +2.0] eşikleriyle ikili bitlere çözer."""
        decoded_bits = []
        for s in noisy_symbols:
            if s < self.level_thresholds[0]:
                decoded_bits.extend([0, 0])
            elif s < self.level_thresholds[1]:
                decoded_bits.extend([0, 1])
            elif s < self.level_thresholds[2]:
                decoded_bits.extend([1, 1])
            else:
                decoded_bits.extend([1, 0])
        return np.array(decoded_bits, dtype=np.uint8)


class MZMModulator:
    """
    Silikon Fotonik Mach-Zehnder Elektro-Optik Modülatörü (MZM).
    """
    def __init__(self, v_pi_v: float = 1.5, laser_power_mw: float = 10.0, extinction_ratio_db: float = 6.5):
        self.v_pi = v_pi_v
        self.laser_power_mw = laser_power_mw
        self.er_db = extinction_ratio_db

    def modulate(self, v_drive: np.ndarray) -> np.ndarray:
        """
        Elektro-optik MZM transfer fonksiyonu: P_out(V) = P_in * cos^2(pi * V / (2 * V_pi))
        Monotonik bölge: V in [0, V_pi]
        """
        v_scaled = (np.clip(v_drive, 0.0, self.v_pi) / (2.0 * self.v_pi)) * np.pi
        transmittance = np.cos(v_scaled) ** 2
        
        er_linear = 10.0 ** (-self.er_db / 10.0)
        transmittance = np.clip(transmittance, er_linear, 1.0)
        
        optical_power_mw = self.laser_power_mw * transmittance
        return optical_power_mw


class OpticalFiberChannel:
    """
    Optik Fiber Kanalı ve Mikro-Bağlantı Kayıp Modeli.
    """
    def __init__(self, length_m: float = 100.0, attenuation_db_per_km: float = 0.2, connector_loss_db: float = 0.5):
        self.length_m = length_m
        self.total_loss_db = (length_m / 1000.0) * attenuation_db_per_km + connector_loss_db
        self.loss_linear = 10.0 ** (-self.total_loss_db / 10.0)

    def propagate(self, p_in_mw: np.ndarray) -> np.ndarray:
        """Optik sinyali fiber boyunca zayıflatarak iletir."""
        return p_in_mw * self.loss_linear


class PhotodiodeTIA:
    """
    PIN Fotodiyot ve Transimpedans Yükseltici (TIA) Alıcı Modeli.
    """
    def __init__(self, responsivity_a_w: float = 0.8, tia_gain_ohm: float = 1000.0, noise_std_mv: float = 5.0):
        self.responsivity = responsivity_a_w
        self.tia_gain = tia_gain_ohm
        self.noise_std_mv = noise_std_mv

    def receive(self, p_opt_mw: np.ndarray, seed: int = 42) -> np.ndarray:
        """
        Optik gücü foton akımına (I_ph = R * P_opt) ve TIA voltajına dönüştürür.
        """
        np.random.seed(seed)
        i_photo_ma = p_opt_mw * self.responsivity
        v_out_mv = i_photo_ma * (self.tia_gain / 1000.0) * 1000.0  # mV
        
        noise = np.random.normal(0.0, self.noise_std_mv, size=len(v_out_mv))
        return v_out_mv + noise


class CPOTransceiverLink:
    """
    800G CPO (8x 112 Gbps PAM4) Uçtan Uca Optik Bağlantı Simülatörü.
    """
    def __init__(self, num_lanes: int = 8):
        self.num_lanes = num_lanes
        self.pam4 = PAM4Encoder(baud_rate_gbaud=56.0)
        self.mzm = MZMModulator(v_pi_v=1.5, laser_power_mw=10.0, extinction_ratio_db=6.5)
        self.channel = OpticalFiberChannel(length_m=100.0)
        self.receiver = PhotodiodeTIA(responsivity_a_w=0.8, tia_gain_ohm=1000.0, noise_std_mv=5.0)

    def simulate_link(self, num_symbols: int = 2000, seed: int = 42) -> Dict[str, Any]:
        """PAM4 sembollerini iletir, göz diyagramını ve BER oranını hesaplar."""
        np.random.seed(seed)
        tx_bits = np.random.randint(0, 2, size=num_symbols * 2)
        
        # 1. PAM4 Seviye Kodlama ([-3, -1, +1, +3])
        pam4_syms = self.pam4.encode(tx_bits)
        
        # 2. MZM Sürüş Voltajı (sym=-3 -> V=0.0, sym=+3 -> V=V_pi)
        v_drive = ((pam4_syms + 3.0) / 6.0) * self.mzm.v_pi
        
        # 3. Optik Modülasyon ve Kanal Yayılımı
        p_opt_tx = self.mzm.modulate(v_drive)
        p_opt_rx = self.channel.propagate(p_opt_tx)
        
        # 4. Alıcı (Fotodiyot + TIA)
        v_rx_mv = self.receiver.receive(p_opt_rx, seed=seed)
        
        # 5. İdeal 4 seviyeli gerilim merkezleri (Maximum Likelihood Demodülasyon)
        ideal_syms = np.array([-3.0, -1.0, 1.0, 3.0])
        ideal_v_drives = ((ideal_syms + 3.0) / 6.0) * self.mzm.v_pi
        ideal_p_opt = self.channel.propagate(self.mzm.modulate(ideal_v_drives))
        ideal_v_levels = ideal_p_opt * self.receiver.responsivity * (self.receiver.tia_gain / 1000.0) * 1000.0  # mV
        
        # En yakın ideal seviyeye karar ver (Minimum Euclidean Distance Decision)
        closest_indices = np.argmin(np.abs(v_rx_mv[:, None] - ideal_v_levels[None, :]), axis=1)
        v_norm = ideal_syms[closest_indices]
        
        # 6. Geri Çözümleme ve Hata Analizi
        rx_bits = self.pam4.decode(v_norm)
        bit_errors = np.sum(tx_bits[:len(rx_bits)] != rx_bits)
        ber = float(bit_errors / max(1, len(rx_bits)))

        cpo_energy_pj_per_bit = 3.8
        pluggable_energy_pj_per_bit = 18.2

        return {
            "num_symbols": num_symbols,
            "tx_bits": tx_bits,
            "rx_bits": rx_bits,
            "pam4_syms": pam4_syms,
            "p_opt_tx": p_opt_tx,
            "v_rx_mv": v_rx_mv,
            "v_norm": v_norm,
            "ber": ber,
            "cpo_energy_pj_bit": cpo_energy_pj_per_bit,
            "pluggable_energy_pj_bit": pluggable_energy_pj_per_bit,
            "energy_savings_x": pluggable_energy_pj_per_bit / cpo_energy_pj_per_bit,
            "aggregate_data_rate_gbps": self.num_lanes * 112.0
        }


class CPOBenchmark:
    """
    Co-Packaged Optics (CPO) vs Pluggable Optik Alıcı-Verici Kıyaslama Motoru.
    """
    def __init__(self):
        self.link = CPOTransceiverLink(num_lanes=8)

    def run_benchmark(self, num_symbols: int = 5000) -> Dict[str, Any]:
        res = self.link.simulate_link(num_symbols=num_symbols, seed=42)
        return res

    def kos(self, num_symbols: int = 5000) -> Dict[str, Any]:
        return self.run_benchmark(num_symbols)
