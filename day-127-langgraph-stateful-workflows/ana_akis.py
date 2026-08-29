"""
Day 127: LangGraph Durumsal Çizge (StateGraph) & Human-in-the-Loop İş Akışları Ana Akışı.
Koşullu dallanma, State Reducer birleştirmesi, bellek kontrol noktaları (Checkpointing) ve insan onayı kesintisi.
"""

import os
import sys
import json

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.cizge_durumu import varsayilan_durum_olustur
from src.is_akislari import iade_akisi_olustur
from src.gorsellestirici import LangGraphGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 127: LangGraph Stateful Workflows & Human-in-the-Loop (StateGraph Engine)")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    cizge = iade_akisi_olustur()

    # -------------------------------------------------------------
    # ADIM 1: Düşük Riskli Otomatik İade İş Akışı
    # -------------------------------------------------------------
    print("\n[1/3] Senaryo 1: Düşük Riskli Otomatik İade (Tutar: 1,850 TL)...")
    durum_dusuk = varsayilan_durum_olustur(musteri_id="MUST_DUSUK_01", talep_tutari=1850.0, baslangic_mesaji="Ürünü iade etmek istiyorum.")
    sonuc_dusuk = cizge.calistir(durum_dusuk)

    print(f"  [✓] Tamamlandı Mı   : {sonuc_dusuk['tamamlandi']}")
    print(f"  [✓] Düğüm Sırası    : {' -> '.join(sonuc_dusuk['durum']['adim_gecmisi'])}")
    print(f"  [✓] Ödeme Yapıldı Mı: {sonuc_dusuk['durum']['odeme_yapildi_mi']}")
    print(f"  [✓] Nihai Durum     : {sonuc_dusuk['durum']['nihai_durum']}")

    # -------------------------------------------------------------
    # ADIM 2: Yüksek Riskli Human-in-the-Loop (HITL) İade İş Akışı
    # -------------------------------------------------------------
    print("\n[2/3] Senaryo 2: Yüksek Riskli İade & İnsan Denetçi Kesintisi (Tutar: 12,500 TL)...")
    durum_yuksek = varsayilan_durum_olustur(musteri_id="MUST_VIP_99", talep_tutari=12500.0, baslangic_mesaji="Büyük tutarlı toplu iade talebi.")

    # 1. Faz: Kesinti Noktasına Kadar Çalıştır
    kesinti_faz = cizge.calistir(durum_yuksek)
    print(f"  [⏸️] 1. FAZ: Çizge Duraklatıldı! Kesinti Noktası: {kesinti_faz['kesinti_noktasi']}")
    print(f"  [⏸️] Durum: {kesinti_faz['durum']['nihai_durum']} | Risk Skoru: %{kesinti_faz['durum']['risk_skoru']*100:.0f}")

    # 2. Faz: İnsan Denetçi Durumu İnceledi ve Onayladı
    print("  [👤] İnsan Denetçi: 'Talep ve fatura incelendi, iadeye ONAY verildi.'")
    nihai_faz = cizge.calistir(kesinti_faz["durum"], insan_onay_yaniti=True)

    print(f"  [✓] 2. FAZ: Çizge Devam Etti ve Tamamlandı!")
    print(f"  [✓] Düğüm Sırası    : {' -> '.join(nihai_faz['durum']['adim_gecmisi'])}")
    print(f"  [✓] Ödeme Durumu    : {nihai_faz['durum']['odeme_yapildi_mi']}")
    print(f"  [✓] Nihai Durum     : {nihai_faz['durum']['nihai_durum']}")

    # -------------------------------------------------------------
    # ADIM 3: Checkpoint Geçmişi ve Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] Kontrol Noktası (Checkpoint / Time Travel) Geçmişi ve Teşhis Panosu Çiziliyor...")
    gecmis = cizge.checkpoint_yoneticisi.gecmis_ozeti()

    print("\n" + "=" * 90)
    print(f"{'ADIM NO':<10} | {'DÜĞÜM ADI':<26} | {'RİSK SKORU':<16} | {'DURUM'}")
    print("-" * 90)
    for k in gecmis:
        print(f"{k['adim']:<10} | {k['dugum']:<26} | %{k['risk']*100:<15.0f} | {k['durum']}")
    print("-" * 90)

    gorsellestirici = LangGraphGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "langgraph_paneli.png")
    gorsellestirici.pano_olustur(
        calisma_sonucu=nihai_faz,
        checkpoint_gecmisi=gecmis,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 105)
    print("[OK] Day 127: LANGGRAPH STATEFUL CYCLIC WORKFLOWS BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
