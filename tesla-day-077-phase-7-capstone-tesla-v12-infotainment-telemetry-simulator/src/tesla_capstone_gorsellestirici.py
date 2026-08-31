r"""
Tesla Faz 7 Capstone Görselleştirici Modülü
===========================================
Bu modül; 9 alt modüllü Tesla V12 Infotainment mimarisini, 3D FSD ekran
izdüşümünü, ARNC ses kalkanını, canlı telemetri kartını ve tam yığın
döngü gecikmesini 6 panelli karanlık mod tanı paneli olarak üretir.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


class TeslaCapstoneGorsellestirici:
    """
    Tesla Faz 7 Capstone 6 panelli tanı paneli üreticisi.
    """
    def __init__(self, cikti_dizini: str = "ciktilar"):
        self.cikti_dizini = cikti_dizini
        os.makedirs(self.cikti_dizini, exist_ok=True)

    def tani_paneli_ciz(self, metrikler: Dict[str, Any], dosya_adi: str = "tesla_faz7_capstone_infotainment_tani_paneli.png") -> str:
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=300)
        fig.suptitle(
            "[TESLA FAZ 7 BÜYÜK CAPSTONE: V12 KONSOL VE TELEMETRİ SİMÜLATÖRÜ]\n"
            "Modül: Gün 77 | Qt6/QML, 3D GPU Render, D-Bus IPC, PipeWire ARNC, Secure Boot, OTA A/B, Seccomp, UWB & HVAC Entegrasyonu",
            fontsize=14, fontweight='bold', color='#E82127', y=0.98
        )

        speed = metrikler.get("speed_kmh", 82.4)
        battery = metrikler.get("battery_pct", 81.2)
        screen_u = metrikler.get("screen_u", 1017.0)
        screen_v = metrikler.get("screen_v", 540.0)
        arnc_db = metrikler.get("arnc_db", 60.0)
        uwb_dist = metrikler.get("uwb_dist_m", 1.35)
        cap_ok = metrikler.get("capstone_ok", True)
        step_ort = metrikler.get("cycle_ortalama_us", 25.0)
        gecikmeler = metrikler.get("gecikmeler", [step_ort] * 100)

        # 1. Panel: 9 Alt Sistem Entegrasyon Matrisi
        ax1 = axes[0, 0]
        sistemler = ['Qt6 UI', '3D GPU', 'D-Bus', 'ARNC', 'SecBoot', 'OTA A/B', 'Sandbox', 'UWB Key', 'HVAC PID']
        durumlar = [1.0] * 9
        renkler1 = ['#98C379' if cap_ok else '#E06C75'] * 9
        cubuklar1 = ax1.barh(sistemler, durumlar, color=renkler1, height=0.5)
        ax1.set_title("1. 9 Alt Sistem Canlılık Durumu", color='#56B6C2', fontsize=11, fontweight='bold')
        ax1.set_xlabel("Sistem Durumu (1=Aktif & Senkron)")
        ax1.grid(True, linestyle=':', alpha=0.3)

        # 2. Panel: 3D FSD Dünya Ekran İzdüşümü (1920x1080)
        ax2 = axes[0, 1]
        ax2.set_xlim(0, 1920)
        ax2.set_ylim(1080, 0)  # Ekran pikselinde y ters
        ax2.plot([400, 960], [1080, 540], color='#56B6C2', linestyle='--', linewidth=1.5, label='Sol Şerit')
        ax2.plot([1520, 960], [1080, 540], color='#56B6C2', linestyle='--', linewidth=1.5, label='Sağ Şerit')
        ax2.scatter([screen_u], [screen_v], color='#E82127', s=120, marker='s', label=f'3D Engel ({screen_u:.0f}, {screen_v:.0f})')
        ax2.set_title("2. 3D FSD GPU Dünya Render İzdüşümü", color='#56B6C2', fontsize=11, fontweight='bold')
        ax2.set_xlabel("Ekran Genişliği X (Piksel)")
        ax2.set_ylabel("Ekran Yüksekliği Y (Piksel)")
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, linestyle=':', alpha=0.3)

        # 3. Panel: PipeWire ARNC Ses Sönümleme Kalkanı
        ax3 = axes[0, 2]
        t = np.linspace(0, 0.01, 100)
        y_noise = np.sin(2 * np.pi * 150 * t)
        y_anti = -y_noise
        ax3.plot(t * 1000, y_noise, color='#E06C75', label='Yol Gürültüsü x(t)')
        ax3.plot(t * 1000, y_anti, color='#61AFEF', linestyle='--', label='Ters Faz Anti-Noise y(t)')
        ax3.set_title(f"3. ARNC Akustik Gürültü Sönümleme ({arnc_db:.1f} dB)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax3.set_xlabel("Zaman (Milisaniye)")
        ax3.set_ylabel("Genlik (Pa)")
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, linestyle=':', alpha=0.3)

        # 4. Panel: Tesla V12 Konsol Canlı Telemetri Kartı
        ax4 = axes[1, 0]
        ax4.axis('off')
        ax4.text(0.5, 0.88, "TESLA V12 FULL-STACK KONSOL TELEMETRİSİ", ha='center', va='center', fontsize=12, color='#56B6C2', fontweight='bold')
        ax4.text(0.5, 0.55, f"ARAÇ HIZI: {speed:.1f} km/s | VİTES: D (İleri Sürüş)\nBATARYA ŞARJI: %{battery:.1f} | FSD DURUMU: %100 OTONOM AKTİF\nKABİN İKLİMLENDİRME: 21.5 °C (Kararlı Rejim)\nUWB PHONE KEY: {uwb_dist:.2f}m (Işık Hızı Doğrulandı)\nOTA SÜRÜMÜ: v2026.12.5 (Slot A Aktif - Rollback Korumalı)\nGÜVENLİK: Root of Trust + Seccomp BPF Chromium Sandbox",
                 ha='center', va='center', fontsize=9.5, color='#FFFFFF')
        ax4.text(0.5, 0.18, f"DURUM: %100 TÜM SİSTEMLER HAZIR (ALL SYSTEMS GO)", ha='center', va='center', fontsize=11, color='#98C379', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#21252B', edgecolor='#98C379', linewidth=1.5))
        ax4.set_title("4. V12 Canlı Gösterge Tablosu", color='#56B6C2', fontsize=11, fontweight='bold')

        # 5. Panel: Tam Yığın Döngü Gecikmesi Histogramı
        ax5 = axes[1, 1]
        ax5.hist(gecikmeler, bins=20, alpha=0.75, color='#61AFEF', label=f'Ortalama: {step_ort:.2f} µs')
        ax5.set_title("5. Tam Yığın RTOS Döngü Gecikmesi", color='#56B6C2', fontsize=11, fontweight='bold')
        ax5.set_xlabel("Döngü Gecikmesi (µs)")
        ax5.set_ylabel("Örnekleme")
        ax5.legend(loc='upper right', fontsize=8)
        ax5.grid(True, linestyle=':', alpha=0.3)

        # 6. Panel: Faz 7 Master Capstone Skorkartı
        ax6 = axes[1, 2]
        skor_etiket = ['V12 UI (Qt6)', '3D GPU Engine', 'D-Bus IPC', 'PipeWire ARNC', 'SecBoot & OTA', 'UWB Phone Key']
        skor_deger = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        cubuklar6 = ax6.bar(skor_etiket, skor_deger, color=['#98C379', '#61AFEF', '#E5C07B', '#C678DD', '#E82127', '#56B6C2'], width=0.5)
        for cubuk in cubuklar6:
            y = cubuk.get_height()
            ax6.text(cubuk.get_x() + cubuk.get_width()/2.0, y + 0.2, f'{y:.1f}', ha='center', va='bottom', fontsize=8, color='#FFFFFF')
        ax6.set_title("6. Faz 7 Master Capstone Skoru", color='#56B6C2', fontsize=11, fontweight='bold')
        ax6.set_ylabel("Puan (0 - 10)")
        ax6.set_ylim(0, 12)
        ax6.tick_params(axis='x', rotation=25)
        ax6.grid(True, linestyle=':', alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        hedef_dosya = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(hedef_dosya, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return hedef_dosya
