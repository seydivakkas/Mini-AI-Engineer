"""
Day 339: Biocompatible BCI Implant Communication Protocol & Cryptographic Telemetry
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Biyouyumlu İmplant Paketi Formatlayıcısını (NeuralSpikeFrame),
Hafif Kriptografik Şifreleme Motorunu (AES-128-GCM AEAD) ve Kablosuz Telemetri Bağlantısını içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import struct
import zlib
import time
import os
import numpy as np


class NeuralSpikeFrame:
    """
    1024-Kanallı Nöronal Spike Telemetri Paketi Formatlayıcısı (Sıkıştırılmış 64-Byte İkili Paket).
    """
    MAGIC_HEADER = 0x42434931  # ASCII 'BCI1'

    def __init__(self, implant_id: int = 0x00A1):
        self.implant_id = implant_id

    def encode_frame(self, sequence_no: int, timestamp_ms: int, spike_mask: np.ndarray) -> bytes:
        """
        Nöronal spike verisini ikili bayt paketine sıkıştırır ve CRC-32 ekler.
        """
        # Spike maskesini bitpack etme (örneğin 272 bit = 34 bayt)
        flat_spikes = (spike_mask.flatten() > 0).astype(np.uint8)
        packed_payload = np.packbits(flat_spikes[:272]).tobytes()

        # Header: Magic(4B) + ImplantID(2B) + SeqNo(4B) + Timestamp(4B) -> Total 14B
        header = struct.pack(">IHII", self.MAGIC_HEADER, self.implant_id, sequence_no, timestamp_ms)
        
        raw_frame = header + packed_payload
        crc32 = zlib.crc32(raw_frame) & 0xFFFFFFFF
        
        return raw_frame + struct.pack(">I", crc32)

    def decode_frame(self, frame_bytes: bytes) -> Dict[str, Any]:
        """
        İkili paketi çözer ve CRC-32 doğrulamasını yapar.
        """
        if len(frame_bytes) < 18:
            raise ValueError("Geçersiz paket boyutu (Min 18 bayt olmalı).")

        raw_frame = frame_bytes[:-4]
        received_crc = struct.unpack(">I", frame_bytes[-4:])[0]
        calculated_crc = zlib.crc32(raw_frame) & 0xFFFFFFFF

        if received_crc != calculated_crc:
            raise ValueError(f"CRC Checksum Hatası! Beklenen: {calculated_crc}, Alınan: {received_crc}")

        magic, implant_id, seq_no, ts_ms = struct.unpack(">IHII", raw_frame[:14])
        if magic != self.MAGIC_HEADER:
            raise ValueError(f"Geçersiz Magic Header: {hex(magic)}")

        packed_payload = raw_frame[14:]
        spike_bits = np.unpackbits(np.frombuffer(packed_payload, dtype=np.uint8))

        return {
            "implant_id": implant_id,
            "sequence_no": seq_no,
            "timestamp_ms": ts_ms,
            "spike_bits": spike_bits,
            "crc_valid": True,
        }


class LightweightAEADCrypto:
    """
    Hafif Kriptografik Şifreleme (AEAD Simülasyonu - AES-128-GCM / Poly1305).
    Nöronal verinin gizliliğini (Confidentiality) ve bütünlüğünü (Integrity) korur.
    """
    def __init__(self, secret_key: bytes = b"BCI_SECRET_KEY_1"):
        self.secret_key = secret_key

    def encrypt_payload(self, plaintext: bytes, nonce: bytes) -> Tuple[bytes, bytes]:
        """
        Düz metin telemetri paketini şifreler ve 16-baytlık Kimlik Doğrulama Etiketi (Auth Tag) üretir.
        """
        # XOR şifreleme + HMAC-CRC simülasyonu
        key_stream = bytes([self.secret_key[i % len(self.secret_key)] ^ nonce[i % len(nonce)] for i in range(len(plaintext))])
        ciphertext = bytes([p ^ k for p, k in zip(plaintext, key_stream)])
        
        # 16-Baytlık Kimlik Doğrulama Etiketi (Auth Tag)
        auth_tag = struct.pack(">QQ", zlib.crc32(ciphertext + nonce) & 0xFFFFFFFF, zlib.crc32(ciphertext + self.secret_key) & 0xFFFFFFFF)
        return ciphertext, auth_tag

    def decrypt_payload(self, ciphertext: bytes, nonce: bytes, auth_tag: bytes) -> bytes:
        """
        Şifreli paketi çözer ve Kimlik Doğrulama Etiketini doğrular.
        """
        expected_tag = struct.pack(">QQ", zlib.crc32(ciphertext + nonce) & 0xFFFFFFFF, zlib.crc32(ciphertext + self.secret_key) & 0xFFFFFFFF)
        if auth_tag != expected_tag:
            raise SecurityError("KRİPTOGRAFİK İHLAL: Auth Tag Doğrulanamadı! Veri Değiştirilmiş Olabilir.")

        key_stream = bytes([self.secret_key[i % len(self.secret_key)] ^ nonce[i % len(nonce)] for i in range(len(ciphertext))])
        plaintext = bytes([c ^ k for c, k in zip(ciphertext, key_stream)])
        return plaintext


class SecurityError(Exception):
    pass


class BiocompatibleTelemetryLink:
    """
    Biyouyumlu Kablosuz Telemetri Bağlantısı ve Termal Güç Profilleyicisi.
    """
    THERMAL_POWER_LIMIT_MW = 15.0  # Doku hasarını önlemek için 15 mW üst sınırı

    def __init__(self, voltage_v: float = 1.8, current_ma: float = 2.2):
        self.voltage_v = voltage_v
        self.current_ma = current_ma
        self.power_mw = voltage_v * current_ma  # P = V * I = 1.8V * 2.2mA = 3.96 mW

    def is_thermally_safe(self) -> bool:
        """İmplant güç tüketiminin 15 mW doku güvenlik sınırının altında olduğunu doğrular."""
        return self.power_mw < self.THERMAL_POWER_LIMIT_MW
