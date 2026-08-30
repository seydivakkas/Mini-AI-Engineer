"""
Day 339: Biocompatible BCI Implant Communication Protocol & Cryptographic Telemetry
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; 64-Byte İkili Telemetri Paket Şifrelemesini (AES-128-GCM AEAD),
Termal Güç Güvenlik Analizini, Tahrifat Saldırı Tespitini ve 6-Panelli Teşhis Panosunu çalıştırır.
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

from src.crypto_telemetry_motoru import (
    NeuralSpikeFrame,
    LightweightAEADCrypto,
    BiocompatibleTelemetryLink,
    SecurityError,
)
from src.telemetry_gorsellestirici import TelemetryGorsellestirici
from src.telemetry_profilleyici import TelemetryProfilleyici


def main():
    print("=" * 75, flush=True)
    print("🔒 DAY 339: Biyouyumlu İmplant İletişim Protokolü ve Kriptografik Telemetri", flush=True)
    print("=" * 75, flush=True)

    np.random.seed(42)

    frame_encoder = NeuralSpikeFrame(implant_id=0x00A1)
    crypto_engine = LightweightAEADCrypto(secret_key=b"NEURALINK_KEY_39")
    telemetry_link = BiocompatibleTelemetryLink(voltage_v=1.8, current_ma=2.2)

    print("\n📌 1) 1024-Kanallı Nöronal Spike Verisi Üretiliyor ve İkili Pakete Sıkıştırılıyor...", flush=True)
    spike_mask = (np.random.rand(1024) > 0.85).astype(np.uint8)

    t0 = time.time()
    raw_frame = frame_encoder.encode_frame(sequence_no=1001, timestamp_ms=1700000000, spike_mask=spike_mask)
    t_encode = (time.time() - t0) * 1000.0

    print(f"✅ Telemetri Paketi İkiliye Sıkıştırıldı | Paket Boyutu: {len(raw_frame)} Bayt | Kodlama Süresi: {t_encode:.4f} ms", flush=True)

    # 2. Kriptografik Şifreleme (AEAD AES-128-GCM)
    print("\n⚡ 2) AEAD Kriptografik Şifreleme (AES-128-GCM / Auth Tag) Uygulanıyor...", flush=True)
    nonce = os.urandom(12)
    t1 = time.time()
    ciphertext, auth_tag = crypto_engine.encrypt_payload(raw_frame, nonce)
    t_crypto = (time.time() - t1) * 1000.0

    print(f"✅ Paket Şifrelendi | Şifreli Metin: {len(ciphertext)} Bayt | Auth Tag: {len(auth_tag)} Bayt | Şifreleme Süresi: {t_crypto:.4f} ms", flush=True)

    # 3. Termal Güç Profillemesi (Doku Hasarı Kontrolü)
    print("\n🌡️ 3) İmplant Biyouyumlu Termal Güç Profillemesi Yapılıyor...", flush=True)
    power_mw = telemetry_link.power_mw
    is_safe = telemetry_link.is_thermally_safe()

    print(f"  • Çalışma Gerilimi: 1.8 V | Çalışma Akımı: 2.2 mA", flush=True)
    print(f"  • Toplam Termal Güç Tüketimi: {power_mw:.2f} mW", flush=True)
    print(f"  • Doku Hasarı Güvenlik Sınırı (< 15 mW): {'✅ GÜVENLİ (Doku Isınması Yok)' if is_safe else '❌ TEHLİKELİ'}", flush=True)

    # 4. Şifre Çözme ve CRC Doğrulaması
    print("\n🔓 4) Alıcı İstasyon Şifre Çözme ve CRC-32 Doğrulaması Yapılıyor...", flush=True)
    decrypted_frame = crypto_engine.decrypt_payload(ciphertext, nonce, auth_tag)
    decoded_info = frame_encoder.decode_frame(decrypted_frame)

    print(f"  • Alınan İmplant ID: {hex(decoded_info['implant_id'])} | Paket Sıra No: {decoded_info['sequence_no']}", flush=True)
    print(f"  • CRC-32 Doğrulaması: ✅ BAŞARILI (Veri Bütünlüğü Korundu)", flush=True)

    # 5. Tahrifat Saldırısı Tespit Testi (Tamper Attack Test)
    print("\n🛡️ 5) Tahrifat & Saldırı Tespit Testi (1-Bit Bozulmuş Paket Enjeksiyonu)...", flush=True)
    tampered_ciphertext = bytes([ciphertext[0] ^ 0x01]) + ciphertext[1:]
    
    tamper_detected = False
    try:
        crypto_engine.decrypt_payload(tampered_ciphertext, nonce, auth_tag)
    except SecurityError as e:
        tamper_detected = True
        print(f"✅ SALDIRI BAŞARIYLA ENGELLENDİ: {e}", flush=True)

    # 6. Profilleme ve Teşhis Panosu
    total_latency_ms = t_encode + t_crypto
    profiler_metrics = TelemetryProfilleyici.profille(
        latency_ms=total_latency_ms,
        thermal_power_mw=power_mw,
        tamper_detection_rate=100.0 if tamper_detected else 0.0
    )

    gorsellestirici = TelemetryGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        plaintext_bytes=raw_frame,
        encrypted_bytes=ciphertext,
        profiler_metrics=profiler_metrics,
        dosya_adi="bci_kripto_telemetri_paneli.png"
    )

    print(f"\n🖼️ 6-Panelli Kriptografik Telemetri Teşhis Grafiği Başarıyla Kaydedildi: [bci_kripto_telemetri_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
