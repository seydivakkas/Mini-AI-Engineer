"""
Day 125: Sandboxed Code Execution & Data Analysis Agent (Code Interpreter) Ana Akışı.
AST güvenlik analizi, izole Python çalıştırma ortamı, saldırı engelleme ve otomatik grafik yakalama.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.guvenlik_denetleyicisi import AstGuvenlikDenetleyicisi
from src.izole_calistirici import IzoleKodCalistirici
from src.veri_analiz_ajani import VeriAnalizAjani
from src.gorsellestirici import SandboxGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 125: Sandboxed Code Execution & Data Analysis Agent (Code Interpreter)")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # ADIM 1: Otonom Veri Analizi Ajanının Çalıştırılması
    # -------------------------------------------------------------
    print("\n[1/3] Otonom Veri Analizi Ajanı (Code Interpreter) Başlatılıyor...")
    ajan = VeriAnalizAjani()

    rapor = ajan.analizi_calistir(
        veri_seti_tanimi="6 Aylık BIST Teknoloji Şirketi Gelir, Maliyet ve Net Kar Verisi",
        analiz_hedefi="Aylık gelir trendini, toplam net karı ve ortalama kar marjını hesaplayıp görselleştir",
        grafik_dizini=cikis_dizini,
    )

    print("\n" + "=" * 90)
    print("                   📊 AJAN TARAFINDAN ÇALIŞTIRILAN SANDBOX ÇIKTISI                      ")
    print("=" * 90)
    print(rapor["stdout"])
    print(f"[-] Çalışma Süresi : {rapor['calisma_suresi_ms']:.2f} ms")
    print(f"[-] Üretilen Grafik : {rapor['grafik_sayisi']} Adet Matplotlib Figürü")
    print("-" * 90)

    # -------------------------------------------------------------
    # ADIM 2: AST Statik Güvenlik ve İhlal Engelleme Gösterimi
    # -------------------------------------------------------------
    print("\n[2/3] AST Statik Güvenlik Denetimi & Zararlı Kod Bloklama Test Ediliyor...")
    zararli_ornekler = [
        ("Sistem Yetkisi Ele Geçirme", "import os\nos.system('rm -rf /')"),
        ("Süreç Başlatma (Subprocess)", "import subprocess\nsubprocess.Popen(['calc.exe'])"),
        ("Dinamik Kod İcrası ve Dosya Okuma", "f = open('/etc/shadow', 'r')\neval('1+1')"),
        ("Sandbox Escape (Kaçış Vektörü)", "alt_siniflar = ().__class__.__bases__[0].__subclasses__()"),
    ]

    print("\n" + "=" * 95)
    print(f"{'SALDIRI SENARYOSU':<35} | {'DURUM':<14} | {'GÜVENLİK İHLALİ'}")
    print("-" * 95)
    for baslik, kod in zararli_ornekler:
        guvenli_mi, ihlaller, skor = AstGuvenlikDenetleyicisi.denetle(kod)
        durum_str = "İZİN VERİLDİ" if guvenli_mi else "ENGELLENDİ"
        ihlal_ozeti = ihlaller[0] if ihlaller else "Temiz"
        print(f"{baslik:<35} | {durum_str:<14} | {ihlal_ozeti}")
    print("-" * 95)

    # -------------------------------------------------------------
    # ADIM 3: Salt LLM vs Sandboxed Interpreter Kıyaslaması ve Teşhis Panosu
    # -------------------------------------------------------------
    print("\n[3/3] Salt LLM vs Sandboxed Interpreter Kıyaslaması ve Teşhis Panosu Çiziliyor...")
    karsilastirma = ajan.benchmark_karsilastir()

    print("\n" + "=" * 90)
    print(f"{'METRİK':<32} | {'SALT LLM METİN (%)':<22} | {'SANDBOX INTERPRETER (%)':<24}")
    print("-" * 90)
    for m, s, i in zip(
        karsilastirma["metrikler"],
        karsilastirma["salt_llm_metin"],
        karsilastirma["sandboxed_interpreter"],
    ):
        print(f"{m:<32} | %{s:>19.1f} | %{i:>21.1f}")
    print("-" * 90)

    gorsellestirici = SandboxGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "sandboxed_agent_paneli.png")
    gorsellestirici.pano_olustur(
        analiz_raporu=rapor,
        karsilastirma=karsilastirma,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 105)
    print("[OK] Day 125: SANDBOXED CODE EXECUTION VE VERİ ANALİZİ AJANI BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
