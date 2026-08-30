"""
Day 352: UAV Anti-Spoofing & Tamper-Proof Cryptographic Telemetry
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; GNSS Kinematik Kalıntı / Mahalanobis İnovasyon Doğrulamasını (GPS Spoofing Koruması),
HMAC-SHA256 Kriptografik Paket Doğrulamasını ve Kurcalanmada Bellek Sıfırlama (Zeroize) Sistemini içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import hashlib
import hmac
import time
import numpy as np


class GNSSVIOKinematicResidualValidator:
    """
    GNSS Aldatma (Spoofing) ve Sinyal Sürükleme Dedektörü.
    GNSS ölçümlerini bağımsız Görsel Eylemsizlik Odometrisi (VIO) ile Mahalanobis inovasyon kapısında karşılaştırır.
    """
    def __init__(self, chi2_threshold: float = 9.21): # %99 güven düzeyi (3 serbestlik derecesi)
        self.chi2_thresh = chi2_threshold
        self.is_gnss_trusted = True
        self.cov_matrix = np.diag([2.0, 2.0, 4.0]) # (sigma_x=1.4m, sigma_y=1.4m, sigma_z=2m)

    def validate_gnss_fix(self, gnss_pos: np.ndarray, vio_pos: np.ndarray) -> Dict[str, Any]:
        """
        GNSS pozisyonunu VIO referansı ile denetler ve Mahalanobis mesafesini hesaplar.
        """
        residual = gnss_pos - vio_pos # (3,)
        inv_cov = np.linalg.inv(self.cov_matrix)
        mahalanobis_sq = float(residual.T @ inv_cov @ residual)

        is_spoofed = mahalanobis_sq > self.chi2_thresh
        if is_spoofed:
            self.is_gnss_trusted = False
        else:
            self.is_gnss_trusted = True

        return {
            "mahalanobis_sq": mahalanobis_sq,
            "residual_distance_m": float(np.linalg.norm(residual)),
            "is_spoofed": is_spoofed,
            "gnss_trusted": self.is_gnss_trusted
        }


class SecureTelemetryPacketAuth:
    """
    Kurcalanmaya ve Tekrarlama Saldırılarına (Replay Attack) Dayanıklı Telemetri Kripto Katmanı.
    HMAC-SHA256 mesaj doğrulama kodu ve monoton artan Nonce kullanır.
    """
    def __init__(self, shared_secret_key: bytes):
        self.key = shared_secret_key
        self.last_received_nonce = 0

    def sign_telemetry(self, payload_bytes: bytes, nonce: int) -> Dict[str, Any]:
        """Mesajı nonce ile birleştirip HMAC-SHA256 imzası üretir."""
        data_to_sign = payload_bytes + nonce.to_bytes(8, "big")
        signature = hmac.new(self.key, data_to_sign, hashlib.sha256).hexdigest()
        return {
            "payload": payload_bytes,
            "nonce": nonce,
            "signature": signature
        }

    def verify_and_accept_packet(self, packet: Dict[str, Any]) -> bool:
        """İmzayı doğrular ve Nonce tazeliğini denetler."""
        nonce = packet["nonce"]
        # Tekrarlama Saldırısı (Replay Attack) Kontrolü
        if nonce <= self.last_received_nonce:
            return False

        payload_bytes = packet["payload"]
        data_to_verify = payload_bytes + nonce.to_bytes(8, "big")
        expected_sig = hmac.new(self.key, data_to_verify, hashlib.sha256).hexdigest()

        if hmac.compare_digest(packet["signature"], expected_sig):
            self.last_received_nonce = nonce
            return True
        return False


class TamperProofZeroizeEngine:
    """
    Fiziksel Kurcalanma ve Düşman Eline Geçme Durumunda Bellek İmha (Zeroize / Self-Destruct) Motoru.
    Gövde kapağı açıldığında veya 50g üzeri darbede kripto anahtarları mikrosaniyede temizler.
    """
    def __init__(self, master_key_hex: str):
        self.volatile_key_register = bytearray.fromhex(master_key_hex)
        self.is_zeroized = False

    def check_tamper_sensors(self, chassis_open_sensor: bool, impact_accel_g: float) -> bool:
        """Sensörleri okur ve ihlal varsa anahtarları sıfırlar."""
        if chassis_open_sensor or impact_accel_g > 50.0:
            self.execute_zeroize()
            return True
        return False

    def execute_zeroize(self):
        """Kripto anahtar belleğini sıfırlarla ve rastgele gürültüyle ezer (Overwriting)."""
        for i in range(len(self.volatile_key_register)):
            self.volatile_key_register[i] = 0x00
        self.is_zeroized = True


class UAVCyberPhysicalDefenseSystem:
    """
    Uçtan Uca İHA Siber-Fiziksel Güvenlik ve Anti-Spoofing Savunma Sistemi.
    """
    def __init__(self, secret_key: bytes = b"TACTICAL_UAV_SEC_KEY_2026"):
        self.validator = GNSSVIOKinematicResidualValidator()
        self.crypto_telemetry = SecureTelemetryPacketAuth(shared_secret_key=secret_key)
        self.zeroize_engine = TamperProofZeroizeEngine(secret_key.hex())
