"""Veri Profilleme Raporu Üreteci - Konsol ve Markdown Çıktıları.

Bu modül; VeriKumesiProfili nesnesini görsel olarak zengin konsol tablolarına
veya GitHub uyumlu Markdown dokümanlarına dönüştürür.
"""

from typing import Dict, Any
from src.veri_profilleyici import VeriKumesiProfili


class ProfilRaporOlusturucu:
    """Veri seti profillerini insan tarafından okunabilir raporlara dönüştürür."""

    @staticmethod
    def konsola_yazdir(profil: VeriKumesiProfili) -> None:
        """Konsola yapılandırılmış, hizalı bir özet raporu basar."""
        cizgi = "=" * 78
        print(f"\n{cizgi}")
        print(">>> OTOMATİK VERİ SETİ PROFİLLEME RAPORU")
        print(cizgi)
        print(f"Toplam Satır Sayısı      : {profil.satir_sayisi:,}")
        print(f"Toplam Sütun Sayısı      : {profil.sutun_sayisi}")
        print(f"Toplam Hücre Adedi       : {profil.toplam_hucre:,}")
        print(f"Toplam Eksik Hücre (NaN) : {profil.toplam_eksik_hucre:,} (%{profil.genel_eksik_orani * 100:.2f})")
        print(f"Bellek Tüketimi          : {profil.toplam_bellek_kb:.2f} KB ({profil.satir_basi_bayt:.1f} bayt/satır)")

        if profil.genel_uyarilar:
            print("\n[!] GENEL KRİTİK UYARILAR:")
            for uyari in profil.genel_uyarilar:
                print(f"    * {uyari}")

        print(f"\n{'-' * 78}")
        print(f"{'Sütun Adı':<20} | {'Fiziksel':<10} | {'Anlamsal Tip':<16} | {'Eksik %':<8} | {'Kardinalite'}")
        print(f"{'-' * 78}")

        for sutun_adi, sp in profil.sutunlar.items():
            print(
                f"{sutun_adi:<20} | {sp.fiziksel_tip:<10} | {sp.anlamsal_tip:<16} | "
                f"%{sp.eksik_orani * 100:<7.1f} | {sp.benzersiz_sayisi:>4} ({sp.kardinalite_orani * 100:.1f}%)"
            )

        print(f"\n{cizgi}")
        print(">>> SAYISAL DAĞILIM VE ŞEKİL PARAMETRELERİ (Çarpıklık & Basıklık)")
        print(cizgi)
        print(f"{'Sütun Adı':<18} | {'Ortalama':<9} | {'Std':<8} | {'Medyan':<8} | {'Çarpıklık':<10} | {'Basıklık'}")
        print(f"{'-' * 78}")

        for sutun_adi, sp in profil.sutunlar.items():
            if sp.istatistikler:
                ist = sp.istatistikler
                print(
                    f"{sutun_adi:<18} | {ist['ortalama']:<9.2f} | {ist['standart_sapma']:<8.2f} | "
                    f"{ist['medyan']:<8.2f} | {ist['carpiklik']:<10.2f} | {ist['basiklik']:.2f}"
                )

        print(f"\n{cizgi}")
        print(">>> SÜTUN BAZLI ALARM VE TAVSİYELER")
        print(cizgi)
        uyari_var_mi = False
        for sutun_adi, sp in profil.sutunlar.items():
            if sp.uyarilar:
                uyari_var_mi = True
                print(f"[*] {sutun_adi}:")
                for u in sp.uyarilar:
                    print(f"    - {u}")

        if not uyari_var_mi:
            print("[V] Harika! Sütunlarda herhangi bir anomali veya yüksek riskli durum tespit edilmedi.")

    @staticmethod
    def markdown_raporu_uret(profil: VeriKumesiProfili) -> str:
        """GitHub ve dokümantasyonlar için Markdown tablosu üretir."""
        md = []
        md.append("# 📊 Veri Seti Profilleme Raporu\n")
        md.append(f"- **Satır / Sütun:** {profil.satir_sayisi} satır, {profil.sutun_sayisi} sütun")
        md.append(f"- **Bellek:** {profil.toplam_bellek_kb} KB")
        md.append(f"- **Genel Eksiklik:** %{profil.genel_eksik_orani * 100:.2f}\n")

        md.append("## 📋 Sütun Özet Tablosu\n")
        md.append("| Sütun | Fiziksel Tip | Anlamsal Tip | Eksik (%) | Benzersiz Değer |")
        md.append("|---|---|---|---|---|")
        for s, sp in profil.sutunlar.items():
            md.append(f"| `{s}` | `{sp.fiziksel_tip}` | {sp.anlamsal_tip} | %{sp.eksik_orani * 100:.1f} | {sp.benzersiz_sayisi} |")

        return "\n".join(md)
